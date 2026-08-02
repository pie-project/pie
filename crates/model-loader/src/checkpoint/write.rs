//! Writing a checkpoint: the output side of `convert`.
//!
//! The one writer the loader has, and `.zt` is the one format it writes.
//! There was a safetensors writer beside this, and what settled it is what the
//! container can say: safetensors describes a tensor as a dtype tag over a
//! shape, so a payload it has no tag for cannot be written at all — which made
//! `convert` refuse every quantized output outside {MXFP4, FP8, INT8}. A `.zt`
//! object carries a *layout*, so the scheme names itself and any scheme the
//! loader can describe can be written down.
//!
//! Three things come free with the format and are the reason it is the only
//! one here:
//!
//! - **Alignment.** Canonical placement puts every tensor on a 64 KiB
//!   boundary, so a written artifact is already streamable. The rewrite that
//!   used to align a safetensors file afterwards — inventing filler tensors to
//!   express a gap the format has no word for — is gone with it.
//! - **Integrity.** Every tensor carries an XXH3 digest and the manifest
//!   carries one of its own, so a cache that rots is an error at load rather
//!   than a wrong answer.
//! - **Provenance.** What the artifact was derived from goes in the file's
//!   attributes instead of a sidecar or a directory name.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ztensor::DType as ZDType;
use ztensor::format::cbor::{self, Value};

use crate::error::Error;
use crate::types::{DType, Encoding, QuantScheme, TensorDecl};

/// One tensor of the file: what to call it, and the bytes as stored.
pub struct WriteTensor<'a> {
    pub decl: &'a TensorDecl,
    pub bytes: &'a [u8],
}

/// The storage type and logical type a declaration's elements are stored as.
///
/// A quantized payload is bytes: the scheme says how to read them, and the
/// layout on the object says which scheme. The one exception is MXFP4's
/// payload, whose elements are half a byte — `f4_e2m1` says so, and the size
/// equation the reader checks follows from it.
fn storage_of(
    decl_dtype: DType,
    encoding: &Encoding,
) -> Result<(ZDType, Option<&'static str>), Error> {
    if let Encoding::Quant(spec) = encoding {
        return Ok(match spec.scheme {
            QuantScheme::Mxfp4E2M1E8M0 => (ZDType::U8, Some("f4_e2m1")),
            _ => (ZDType::U8, None),
        });
    }
    Ok(match decl_dtype {
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
    })
}

/// The layout profile a declaration's encoding names, and the attributes
/// that make it readable again.
///
/// `zt.quant_group/1` is parametric: what distinguishes AWQ from GPTQ from
/// MLX-affine is the packing order, the scale form and the zero-point form,
/// so those are written down and the scheme's *name* is not. A reader
/// recovers the scheme by looking at the parameters, which is what keeps the
/// file readable by something that never heard of pie's enum.
fn profile_of(encoding: &Encoding) -> Result<(&'static str, Option<Value>), Error> {
    let Encoding::Quant(spec) = encoding else {
        return Ok(("dense", None));
    };
    let spec = spec.clone().normalized();
    let bits = u64::from(spec.normalized_bits());
    let group = u64::from(spec.normalized_group_size());
    let axis = u64::from(spec.channel_axis.map(|a| a.0).unwrap_or(0));

    // The GGUF family is opaque: the block struct interleaves its scales with
    // its codes, so the profile names the layout and carries the constants a
    // reader needs to check sizes.
    if let Some((elems, bytes)) = spec.block_layout() {
        let name = match spec.scheme {
            QuantScheme::GgufQ4_0 => "gguf.q4_0/1",
            QuantScheme::GgufQ4K => "gguf.q4_k/1",
            QuantScheme::GgufQ5_0 => "gguf.q5_0/1",
            QuantScheme::GgufQ5K => "gguf.q5_k/1",
            QuantScheme::GgufQ8_0 => "gguf.q8_0/1",
            other => {
                return Err(Error::Checkpoint(format!(
                    "{other:?} reports a block layout but has no gguf profile"
                )));
            }
        };
        return Ok((
            name,
            Some(cbor::map([
                ("elems_per_block", elems),
                ("block_bytes", bytes),
            ])),
        ));
    }

    match spec.scheme {
        QuantScheme::None => Ok(("dense", None)),

        // OCP Microscaling: its own profile, because the element is a
        // sub-byte float and the scale form is fixed by that specification.
        QuantScheme::Mxfp4E2M1E8M0 => Ok((
            "zt.mx/1",
            Some(cbor::map([("axis", axis), ("block_size", group)])),
        )),

        // FP8 weights are not group-quantized codes: they are plain f8
        // elements whose scales, when there are any, are a separate tensor a
        // contract pairs with them. `dense` plus the logical type says that
        // exactly.
        QuantScheme::Fp8E4M3 | QuantScheme::Fp8E5M2 => Ok(("dense", None)),

        // Everything else is a point in the affine-group space.
        scheme => {
            let (order, zero) = match scheme {
                QuantScheme::AwqInt4 => ("lsb_first", zero_tensor("same_as_data")),
                QuantScheme::GptqInt4 => ("msb_first", zero_tensor("same_as_data")),
                QuantScheme::MlxAffineU4 => ("lsb_first", zero_tensor("plain")),
                QuantScheme::Int4B8 => ("lsb_first", zero_implied(8)),
                QuantScheme::Int8Symmetric => ("lsb_first", zero_none()),
                QuantScheme::Int8Asymmetric => ("lsb_first", zero_tensor("plain")),
                other => {
                    return Err(Error::Checkpoint(format!(
                        "{other:?} has no zTensor layout profile"
                    )));
                }
            };
            let word = if bits == 8 { "u8" } else { "u32" };
            let word_bits = if bits == 8 { 8 } else { 32 };
            let per_word = word_bits / bits.max(1);
            let scale_form = match scheme {
                QuantScheme::MlxAffineU4 | QuantScheme::AwqInt4 | QuantScheme::GptqInt4 => {
                    "f16_factors"
                }
                _ => "f32_factors",
            };
            Ok((
                "zt.quant_group/1",
                Some(cbor::map([
                    ("axis", Value::from(axis)),
                    ("bits", Value::from(bits)),
                    ("group_size", Value::from(group)),
                    (
                        "packing",
                        cbor::map([
                            ("order", Value::from(order)),
                            ("per_word", Value::from(per_word)),
                            ("word", Value::from(word)),
                        ]),
                    ),
                    ("scale_form", Value::from(scale_form)),
                    ("zero_point", zero),
                ])),
            ))
        }
    }
}

fn zero_none() -> Value {
    cbor::map([("form", "none")])
}

fn zero_implied(value: u64) -> Value {
    cbor::map([
        ("form", Value::from("implied")),
        ("value", Value::from(value)),
    ])
}

fn zero_tensor(packing: &str) -> Value {
    cbor::map([("form", "tensor"), ("packing", packing)])
}

/// Writes a checkpoint one tensor at a time, payloads in chunks.
///
/// The streaming face of [`write_zt`], for callers whose payloads should never
/// all be resident at once — `pie model convert` copies multi-gigabyte
/// tensors straight from the source checkpoint through a bounded buffer.
/// Canonical form requires objects in ascending name order; [`write_zt`] sorts
/// for its caller, this type trusts its caller to add in order.
pub struct CheckpointWriter {
    /// `None` only after [`finish`](CheckpointWriter::finish) has taken it.
    writer: Option<ztensor::Writer>,
    open: Option<ztensor::Sink>,
    /// Present when the output is a shard set rather than one file.
    sharding: Option<Sharding>,
}

/// The state of a multi-file write: where the root goes, which shard is open,
/// and what the root will have to say about them.
///
/// The shape follows from one decision — **a shard is an ordinary `.zt`**, not
/// a bare blob heap. zTensor offers both (spec §7.2), and the bare kind
/// (`DataShardWriter`) takes each blob as one `&[u8]`, which would mean
/// holding a whole tensor in memory. Converting a checkpoint larger than
/// memory is a property this writer exists to have, so shards are written
/// through the same streaming path single-file artifacts use, and the root
/// references their bytes with [`Writer::link`] — which copies nothing and
/// carries each part's digest across.
///
/// Three things fall out of that and are worth having: every shard verifies on
/// its own, every shard is openable on its own, and the root stays small
/// enough to be cheap to read.
struct Sharding {
    root: PathBuf,
    attributes: BTreeMap<String, String>,
    /// Soft cap. A tensor is never split, so a shard closes *after* the tensor
    /// that crossed the line — and one tensor larger than the cap gets a shard
    /// to itself rather than an error.
    max_bytes: u64,
    /// Bytes declared into the shard currently open.
    current_bytes: u64,
    /// 1-based index of the shard currently open.
    index: u32,
    /// Shards already published, in order: their table names and paths.
    done: Vec<(String, PathBuf)>,
    /// Metadata objects, held back for the root.
    ///
    /// They belong there rather than in a shard: opening one file has to
    /// answer "what model is this", and metadata that moved with the weights
    /// would make that depend on which shard you happened to open.
    meta: Vec<(String, Vec<u8>)>,
}

/// Writes one `dense` `u8` metadata object. Shared by the single-file path and
/// the root of a shard set, so the two cannot describe metadata differently.
fn write_meta_object(writer: &mut ztensor::Writer, name: &str, bytes: &[u8]) -> Result<(), Error> {
    writer
        .object(name, |o| {
            o.shape(vec![bytes.len() as u64])
                .layout("dense")
                .part("data", |p| p.dtype(ZDType::U8).bytes(bytes))
        })
        .map_err(Error::from)
}

/// The shard-table name for a 1-based index, and the file it resolves to.
///
/// Five digits, matching zTensor's positional convention (Appendix B): the
/// table holds `00001` and the resolver looks beside the root for
/// `<stem>-00001.zt`. The table never holds a path — a name is all a shard
/// reference carries, because a file name is not something the format can
/// verify.
fn shard_name(index: u32) -> String {
    format!("{index:05}")
}

fn shard_path(root: &Path, index: u32) -> PathBuf {
    let stem = root
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    root.parent()
        .unwrap_or(Path::new("."))
        .join(format!("{stem}-{}.zt", shard_name(index)))
}

impl CheckpointWriter {
    /// Opens a checkpoint at `path`; `metadata` lands in the file's
    /// attributes.
    ///
    /// Publication is atomic: the writer puts bytes beside the target and
    /// moves them into place on [`finish`](Self::finish), so a run that dies
    /// mid-write leaves nothing.
    pub fn create(path: &Path, metadata: &BTreeMap<String, String>) -> Result<Self, Error> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                Error::Checkpoint(format!("cannot create {}: {err}", parent.display()))
            })?;
        }
        let mut writer = ztensor::Writer::publish(path).map_err(Error::from)?;
        if !metadata.is_empty() {
            writer.set_attributes(Value::Map(
                metadata
                    .iter()
                    .map(|(k, v)| (Value::Text(k.clone()), Value::Text(v.clone())))
                    .collect(),
            ));
        }
        Ok(Self {
            writer: Some(writer),
            open: None,
            sharding: None,
        })
    }

    /// Opens a checkpoint that spills into shards once one file passes
    /// `max_shard_bytes`.
    ///
    /// The output is a root `.zt` beside `<stem>-00001.zt`, `<stem>-00002.zt`,
    /// … — zTensor's native multi-file model (spec §7), not a convention laid
    /// over single files. The root carries the manifest, the metadata and a
    /// shard table naming each shard by size and digest; the shards carry the
    /// weights.
    ///
    /// The cap is a *soft* one. A tensor is never split across shards, so a
    /// shard closes after whichever tensor crossed the line, and a tensor
    /// bigger than the whole cap gets a shard to itself.
    pub fn create_sharded(
        root: &Path,
        metadata: &BTreeMap<String, String>,
        max_shard_bytes: u64,
    ) -> Result<Self, Error> {
        if max_shard_bytes == 0 {
            return Err(Error::Checkpoint(
                "a shard size of zero would put every tensor in its own file".into(),
            ));
        }
        if let Some(parent) = root.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                Error::Checkpoint(format!("cannot create {}: {err}", parent.display()))
            })?;
        }
        let first = shard_path(root, 1);
        let writer = ztensor::Writer::publish(&first).map_err(Error::from)?;
        Ok(Self {
            writer: Some(writer),
            open: None,
            sharding: Some(Sharding {
                root: root.to_path_buf(),
                attributes: metadata.clone(),
                max_bytes: max_shard_bytes,
                current_bytes: 0,
                index: 1,
                done: Vec::new(),
                meta: Vec::new(),
            }),
        })
    }

    /// Declares a tensor and opens it for writing. Its payload is exactly
    /// `nbytes` bytes, delivered by [`write`](Self::write).
    pub fn begin_tensor(&mut self, decl: &TensorDecl, nbytes: u64) -> Result<(), Error> {
        crate::checkpoint::meta::reject_reserved(&decl.name)?;
        if self.open.is_some() {
            return Err(Error::Checkpoint(format!(
                "tensor {} was begun while another is still open",
                decl.name
            )));
        }
        self.roll_if_full(nbytes)?;
        let (dtype, ltype) = storage_of(decl.encoding.dtype(), &decl.encoding)?;
        let shape: Vec<u64> = decl
            .shape
            .iter()
            .map(|&d| {
                u64::try_from(d).map_err(|_| {
                    Error::Checkpoint(format!("tensor {} has negative extent {d}", decl.name))
                })
            })
            .collect::<Result<_, _>>()?;
        let (layout, attributes) = profile_of(&decl.encoding)?;
        let sink = self
            .writer()
            .stream(&decl.name, |mut o| {
                o = o.shape(shape).layout(layout);
                if let Some(attributes) = attributes {
                    o = o.attributes(attributes);
                }
                o.part("data", |mut p| {
                    p = p.dtype(dtype);
                    if let Some(ltype) = ltype {
                        p = p.logical(ltype);
                    }
                    p.length(nbytes)
                })
            })
            .map_err(Error::from)?;
        self.open = Some(sink);
        if let Some(sharding) = &mut self.sharding {
            sharding.current_bytes = sharding.current_bytes.saturating_add(nbytes);
        }
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

    /// Adds a tensor whose payload is already in memory.
    pub fn add_tensor(&mut self, decl: &TensorDecl, bytes: &[u8]) -> Result<(), Error> {
        self.begin_tensor(decl, bytes.len() as u64)?;
        self.write(bytes)?;
        self.end_tensor()
    }

    /// Adds a metadata object at `path` under the reserved namespace.
    ///
    /// `path` is the name *without* the prefix — `"tokenizer/vocab_bytes"`,
    /// `"model/descriptor"` — so a caller cannot half-qualify a name and land
    /// outside the namespace it meant to write into.
    ///
    /// The payload is stored as a `dense` `u8` object because zTensor has no
    /// non-tensor object; it therefore gets the same alignment and the same
    /// per-object digest as a weight, which is the point — metadata that
    /// versions with the weights under one manifest is what the artifact
    /// exists to give. Callers must still add in ascending name order together
    /// with the weights; the namespace is not written as a block of its own.
    pub fn add_meta(&mut self, path: &str, bytes: &[u8]) -> Result<(), Error> {
        if self.open.is_some() {
            return Err(Error::Checkpoint(format!(
                "metadata object {path} was added while a tensor is still open"
            )));
        }
        // In a shard set the metadata belongs to the root, so it is held here
        // rather than written into whichever shard happens to be open.
        if let Some(sharding) = &mut self.sharding {
            sharding.meta.push((path.to_string(), bytes.to_vec()));
            return Ok(());
        }
        let name = crate::checkpoint::meta::meta_name(path);
        write_meta_object(self.writer(), &name, bytes)
    }

    /// Closes the open shard and opens the next, when the one open is full.
    ///
    /// Called before a tensor is declared, so the decision is "does this
    /// tensor still fit" rather than "did the last one overflow" — which is
    /// what keeps a tensor whole. An empty shard never rolls, so a tensor
    /// larger than the cap lands in one of its own instead of in a file that
    /// would be empty otherwise.
    fn roll_if_full(&mut self, nbytes: u64) -> Result<(), Error> {
        let Some(sharding) = &self.sharding else {
            return Ok(());
        };
        if sharding.current_bytes == 0
            || sharding.current_bytes.saturating_add(nbytes) <= sharding.max_bytes
        {
            return Ok(());
        }
        let writer = self.writer.take().expect("writer present");
        writer.finish().map_err(Error::from)?;

        let sharding = self.sharding.as_mut().expect("sharding present");
        sharding.done.push((
            shard_name(sharding.index),
            shard_path(&sharding.root, sharding.index),
        ));
        sharding.index += 1;
        sharding.current_bytes = 0;
        let next = shard_path(&sharding.root, sharding.index);
        self.writer = Some(ztensor::Writer::publish(&next).map_err(Error::from)?);
        Ok(())
    }

    /// Writes the root of a shard set: the table, the links, the metadata.
    ///
    /// Non-canonical, because canonical form is single-file by definition
    /// (spec §6.3 rule 6, which the spec calls deferred rather than
    /// impossible). What that costs is only the *label*: since zTensor 2.1.0 a
    /// non-canonical writer still places on 64 KiB, `link_shard` carries every
    /// part's digest across, and the names ascend because the caller adds in
    /// that order. Nothing on disk records the label and no reader checks it.
    fn finish_sharded(mut self, sharding: Sharding) -> Result<(), Error> {
        let writer = self.writer.take().expect("writer present");
        writer.finish().map_err(Error::from)?;
        let mut shards = sharding.done;
        shards.push((
            shard_name(sharding.index),
            shard_path(&sharding.root, sharding.index),
        ));

        let mut root = ztensor::Writer::options()
            .canonical(false)
            .publish(&sharding.root)
            .map_err(Error::from)?;
        if !sharding.attributes.is_empty() {
            root.set_attributes(Value::Map(
                sharding
                    .attributes
                    .iter()
                    .map(|(k, v)| (Value::Text(k.clone()), Value::Text(v.clone())))
                    .collect(),
            ));
        }
        for (name, path) in &shards {
            // Registration and one link per tensor, in the one order that
            // works. zTensor briefly had `Writer::link_shard` for exactly this
            // and it was dropped again in `1d28826`; when it comes back these
            // four steps collapse to one call.
            //
            // The identity is read back from the finished shard rather than
            // accumulated while writing: it is a digest of the bytes that are
            // actually there, which is the only version of it worth having.
            // XXH3 rather than SHA-256 because an artifact in the local model
            // store is neither signed nor distributed, and this is the digest
            // every tensor in it already carries (§6.5).
            let identity = ztensor::read::shard_identity(path, ztensor::DigestAlgorithm::Xxh3)
                .map_err(Error::from)?;
            root.add_shard(name.clone(), &identity)
                .map_err(Error::from)?;
            let manifest = ztensor::read::manifest_of(path)
                .map_err(Error::from)?
                .ok_or_else(|| {
                    Error::Checkpoint(format!("shard {} carries no manifest", path.display()))
                })?;
            for (object_name, object) in &manifest.objects {
                root.link(object_name.clone(), object, name)
                    .map_err(Error::from)?;
            }
        }
        for (path, bytes) in &sharding.meta {
            let name = crate::checkpoint::meta::meta_name(path);
            write_meta_object(&mut root, &name, bytes)?;
        }
        root.finish().map_err(Error::from)?;
        Ok(())
    }

    /// Closes the manifest and moves the file into place.
    pub fn finish(mut self) -> Result<(), Error> {
        if self.open.is_some() {
            return Err(Error::Checkpoint(
                "finish was called while a tensor is still open".into(),
            ));
        }
        if let Some(sharding) = self.sharding.take() {
            return self.finish_sharded(sharding);
        }
        let writer = self.writer.take().expect("writer present");
        writer.finish().map_err(Error::from)?;
        Ok(())
    }

    fn writer(&mut self) -> &mut ztensor::Writer {
        self.writer.as_mut().expect("writer present")
    }
}

/// Writes `tensors` as one canonical `.zt` file at `path`.
///
/// `metadata` lands in the file's attributes. Canonical form requires
/// ascending names, so the tensors are sorted on the way out — which also
/// makes the output byte-identical for identical input.
pub fn write_zt(
    path: &Path,
    metadata: &BTreeMap<String, String>,
    tensors: &[WriteTensor<'_>],
) -> Result<(), Error> {
    let mut writer = CheckpointWriter::create(path, metadata)?;
    let mut ordered: Vec<&WriteTensor<'_>> = tensors.iter().collect();
    ordered.sort_by(|a, b| a.decl.name.cmp(&b.decl.name));
    for tensor in ordered {
        writer.add_tensor(tensor.decl, tensor.bytes)?;
    }
    writer.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::zt::parse_checkpoint;
    use crate::types::{QuantSpec, TensorDecl, TensorId, Visibility};

    fn decl(name: &str, shape: Vec<i64>, encoding: Encoding) -> TensorDecl {
        TensorDecl {
            id: TensorId(0),
            name: name.to_string(),
            shape,
            encoding,
            alignment: 64,
            visibility: Visibility::default(),
        }
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("zt_write_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A metadata object survives the round trip, and every weight-facing view
    /// of the artifact leaves it out: it is not a weight, not a materialization
    /// input, and not resolvable by a contract.
    ///
    /// This is the whole reason the namespace exists. zTensor has no non-tensor
    /// object, so `__meta__/tokenizer/vocab_bytes` is on disk a `dense` `u8`
    /// object — byte-identical in kind to a `u8` weight. If any of these views
    /// stopped filtering, a tokenizer vocab would be planned, copied into a
    /// contract, or uploaded to a device as if it were a tensor.
    #[test]
    fn a_metadata_object_round_trips_and_stays_out_of_the_weight_paths() {
        use crate::checkpoint::meta;
        use crate::contract::infer::CheckpointTypes;

        let dir = tmpdir("meta");
        let path = dir.join("model.zt");
        let weight: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let descriptor = br#"{"arch":"llama3","hidden_size":64}"#;
        let vocab = b"<s></s>hello world";

        let mut writer = CheckpointWriter::create(&path, &BTreeMap::new()).unwrap();
        // Ascending name order across weights *and* metadata: `__meta__/` (0x5F)
        // sorts before a lowercase weight name, so the namespace leads here.
        writer.add_meta("model/descriptor", descriptor).unwrap();
        writer.add_meta("tokenizer/vocab_bytes", vocab).unwrap();
        writer
            .add_tensor(
                &decl("model.embed.weight", vec![4, 4], Encoding::Raw(DType::F32)),
                &weight,
            )
            .unwrap();
        writer.finish().unwrap();

        let parsed = parse_checkpoint(&path).unwrap();

        // All three objects are in the manifest…
        assert_eq!(parsed.tensors.len(), 3);
        // …but only one of them is a weight.
        let weights: Vec<&str> = parsed.weights().map(|t| t.name.as_str()).collect();
        assert_eq!(weights, ["model.embed.weight"]);
        let objects: Vec<&str> = parsed.meta_objects().map(|t| t.name.as_str()).collect();
        assert_eq!(
            objects,
            [
                "__meta__/model/descriptor",
                "__meta__/tokenizer/vocab_bytes"
            ]
        );

        // Addressable by path, and the bytes are exactly what went in.
        let found = parsed.meta_object("tokenizer/vocab_bytes").unwrap();
        assert_eq!(found.span_bytes, vocab.len() as u64);
        let file = &parsed.files[found.file_id.0 as usize];
        let raw = std::fs::read(&file.path).unwrap();
        let at = found.file_offset as usize;
        assert_eq!(&raw[at..at + vocab.len()], vocab);

        // The materialization split is total, and metadata is its own set —
        // never decoded, never copied as if it were a weight.
        let materialization = crate::contract::materialize::materialize_contract(&parsed).unwrap();
        // The weight is F32, which normalizes to BF16 rather than passing
        // through; which set it lands in is not this test's subject, only that
        // it is a weight's set and never metadata's.
        assert_eq!(materialization.decoded, ["model.embed.weight"]);
        assert!(materialization.passthrough.is_empty());
        assert_eq!(
            materialization.meta,
            [
                "__meta__/model/descriptor",
                "__meta__/tokenizer/vocab_bytes"
            ]
        );

        // A contract cannot name a metadata object: the reserved namespace is
        // absent from the type environment a contract resolves against.
        let sources = crate::checkpoint::Sources::new(&parsed);
        assert!(sources.by_name("model.embed.weight").is_some());
        assert!(sources.by_name("__meta__/tokenizer/vocab_bytes").is_none());
        assert!(
            parsed
                .tensor_type("__meta__/tokenizer/vocab_bytes")
                .is_none()
        );

        // And a weight cannot be written into the namespace at all.
        let mut writer = CheckpointWriter::create(&dir.join("bad.zt"), &BTreeMap::new()).unwrap();
        let err = writer
            .add_tensor(
                &decl(
                    &meta::meta_name("smuggled"),
                    vec![4],
                    Encoding::Raw(DType::U8),
                ),
                &[0u8; 4],
            )
            .unwrap_err();
        assert!(err.to_string().contains("reserved metadata namespace"));

        std::fs::remove_dir_all(dir).ok();
    }

    /// A sharded artifact holds the same model as a single-file one, and the
    /// reader cannot tell which it opened.
    ///
    /// That is the whole claim of §6: sharding is the format's native
    /// multi-file model, not a convention laid over single files, so
    /// everything downstream — names, bytes, digests, metadata — is unchanged.
    /// Reading goes through the root, whose shard table names each shard by
    /// size and digest; the files themselves are found beside it.
    #[test]
    fn a_sharded_artifact_reads_back_as_one_model() {
        use crate::checkpoint::meta;

        let dir = tmpdir("sharded");
        let payloads: Vec<Vec<u8>> = (0..6u8)
            .map(|i| vec![i.wrapping_mul(37).wrapping_add(1); 40_000])
            .collect();
        let decls: Vec<TensorDecl> = (0..6)
            .map(|i| decl(&format!("w{i:02}"), vec![40_000], Encoding::Raw(DType::U8)))
            .collect();
        let descriptor = br#"{"arch":"llama3"}"#;

        let write = |path: &std::path::Path, sharded: bool| {
            let mut writer = if sharded {
                // 100 KB against 40 KB tensors: two per shard, three shards.
                CheckpointWriter::create_sharded(path, &BTreeMap::new(), 100_000).unwrap()
            } else {
                CheckpointWriter::create(path, &BTreeMap::new()).unwrap()
            };
            writer.add_meta("model/descriptor", descriptor).unwrap();
            for (decl, bytes) in decls.iter().zip(&payloads) {
                writer.add_tensor(decl, bytes).unwrap();
            }
            writer.finish().unwrap();
        };

        let one = dir.join("single.zt");
        let many = dir.join("sharded.zt");
        write(&one, false);
        write(&many, true);

        // The shards are beside the root under the positional convention, and
        // the root is small: it holds the manifest and the metadata, no
        // weights.
        for i in 1..=3 {
            let shard = dir.join(format!("sharded-{i:05}.zt"));
            assert!(shard.is_file(), "{} is missing", shard.display());
        }
        assert!(!dir.join("sharded-00004.zt").exists(), "one shard too many");

        // Same objects, same order, same bytes — through the root.
        let single = parse_checkpoint(&one).unwrap();
        let sharded = parse_checkpoint(&many).unwrap();
        let names = |cp: &crate::checkpoint::CheckpointMetadata| -> Vec<String> {
            cp.tensors.iter().map(|t| t.name.clone()).collect()
        };
        assert_eq!(names(&sharded), names(&single));
        assert_eq!(sharded.weights().count(), 6);
        assert_eq!(sharded.meta_objects().count(), 1);

        // The weights really do live in more than one file, and none of them
        // in the root: `link` moves no bytes, so the root stays a manifest and
        // its metadata. (Its size is dominated by 64 KiB padding, not by
        // payload, which is why this asks where the bytes are rather than how
        // many there are.)
        let files: std::collections::BTreeSet<u32> =
            sharded.weights().map(|t| t.file_id.0).collect();
        assert_eq!(files.len(), 3, "the weights did not spread across shards");
        assert!(
            !files.contains(&0),
            "a weight was written into the root instead of a shard"
        );
        assert_eq!(sharded.files[0].path, many.display().to_string());

        for tensor in sharded.weights() {
            let file = &sharded.files[tensor.file_id.0 as usize];
            let raw = std::fs::read(&file.path).unwrap();
            let at = tensor.file_offset as usize;
            let got = &raw[at..at + tensor.span_bytes as usize];
            let want = &payloads[tensor.name[1..].parse::<usize>().unwrap()];
            assert_eq!(got, want.as_slice(), "{}", tensor.name);
            // 64 KiB placement survives the non-canonical root: it is asked
            // for explicitly rather than inherited from canonical form.
            assert_eq!(
                tensor.file_offset % 65536,
                0,
                "{} is not page-placed",
                tensor.name
            );
        }

        // Metadata is in the root, so opening one file identifies the model.
        let found = sharded.meta_object("model/descriptor").unwrap();
        assert_eq!(
            sharded.files[found.file_id.0 as usize].path,
            many.display().to_string()
        );
        assert!(meta::is_meta(&found.name));

        // Every digest verifies, across the shard boundary.
        assert_eq!(
            crate::checkpoint::zt::verify_checkpoint(&many).unwrap(),
            7,
            "the root does not verify its shards' tensors"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// A tensor is never split, so a tensor larger than the cap gets a shard
    /// of its own rather than an error or a straddling blob.
    /// Every payload of a shard set lands on a 64 KiB boundary — in the
    /// shards *and* in the root.
    ///
    /// A shard set cannot be canonical (§6.3 rule 6), and until zTensor 2.1.0
    /// leaving canonical form also dropped placement to the 4 KiB floor, so
    /// this writer asked for `ALIGN_CANONICAL` back by hand. 2.1.0 made 64 KiB
    /// the default for non-canonical writers too and the explicit request went
    /// away — which means the property now rests on a default in another
    /// crate. It is the property the artifact exists for (per-tensor page
    /// exclusivity is what lets the driver mmap-stream routed experts), and a
    /// regression in it would be invisible: every file still reads back fine.
    /// So it is checked here rather than assumed.
    #[test]
    fn a_shard_set_places_on_64_kib() {
        let dir = tmpdir("shard-align");
        let root = dir.join("model.zt");
        let mut writer =
            CheckpointWriter::create_sharded(&root, &BTreeMap::new(), 100_000).unwrap();
        writer
            .add_meta("model/descriptor", br#"{"arch":"llama3"}"#)
            .unwrap();
        for i in 0..6 {
            let d = decl(&format!("w{i:02}"), vec![40_000], Encoding::Raw(DType::U8));
            writer.add_tensor(&d, &vec![i as u8 + 1; 40_000]).unwrap();
        }
        writer.finish().unwrap();

        let mut checked = 0usize;
        let mut files = vec![root.clone()];
        files.extend((1..=3).map(|i| dir.join(format!("model-{i:05}.zt"))));
        for file in &files {
            let manifest = ztensor::read::manifest_of(file)
                .unwrap()
                .expect("a manifest");
            for (name, object) in &manifest.objects {
                for (part, blob) in &object.parts {
                    assert_eq!(
                        blob.blob.offset % ztensor::format::ALIGN_CANONICAL,
                        0,
                        "{}: {name}/{part} at {} is not on a {} boundary",
                        file.display(),
                        blob.blob.offset,
                        ztensor::format::ALIGN_CANONICAL,
                    );
                    checked += 1;
                }
            }
        }
        // Six weights, once in a shard and once linked from the root, plus the
        // root's metadata object.
        assert_eq!(checked, 13, "expected every payload to be checked");
    }

    #[test]
    fn a_tensor_larger_than_the_cap_gets_its_own_shard() {
        let dir = tmpdir("shard_oversize");
        let root = dir.join("model.zt");
        let small = vec![1u8; 1_000];
        let huge = vec![2u8; 50_000];

        let mut writer = CheckpointWriter::create_sharded(&root, &BTreeMap::new(), 8_000).unwrap();
        writer
            .add_tensor(&decl("a", vec![1_000], Encoding::Raw(DType::U8)), &small)
            .unwrap();
        writer
            .add_tensor(&decl("b", vec![50_000], Encoding::Raw(DType::U8)), &huge)
            .unwrap();
        writer
            .add_tensor(&decl("c", vec![1_000], Encoding::Raw(DType::U8)), &small)
            .unwrap();
        writer.finish().unwrap();

        let parsed = parse_checkpoint(&root).unwrap();
        assert_eq!(parsed.weights().count(), 3);
        // `a` fills shard 1; `b` does not fit beside it so shard 2 is its own;
        // `c` does not fit beside `b` either, so shard 3. Plus the root, which
        // `files` counts as the checkpoint's first file.
        assert_eq!(parsed.files.len(), 4);
        for tensor in parsed.weights() {
            let file = &parsed.files[tensor.file_id.0 as usize];
            let raw = std::fs::read(&file.path).unwrap();
            let at = tensor.file_offset as usize;
            let want = if tensor.name == "b" { &huge } else { &small };
            assert_eq!(&raw[at..at + tensor.span_bytes as usize], want.as_slice());
        }
        std::fs::remove_dir_all(dir).ok();
    }

    /// Provenance written into file attributes reads back.
    #[test]
    fn file_attributes_round_trip_as_provenance() {
        let dir = tmpdir("provenance");
        let path = dir.join("model.zt");
        let mut attributes = BTreeMap::new();
        attributes.insert("pie_source_repo".to_string(), "qwen/qwen3-0.6b".to_string());
        attributes.insert("pie_source_revision".to_string(), "abc123".to_string());

        let da = decl("w", vec![4], Encoding::Raw(DType::U8));
        write_zt(
            &path,
            &attributes,
            &[WriteTensor {
                decl: &da,
                bytes: &[1u8, 2, 3, 4],
            }],
        )
        .unwrap();

        assert_eq!(
            crate::checkpoint::zt::read_attributes(&path).unwrap(),
            attributes
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dense_tensors_round_trip_through_the_reader() {
        let dir = tmpdir("dense");
        let path = dir.join("model.zt");
        let a: Vec<u8> = (0..32u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let b = vec![7u8; 16];
        let da = decl("a.weight", vec![8, 4], Encoding::Raw(DType::F32));
        let db = decl("b.mask", vec![16], Encoding::Raw(DType::U8));
        let mut meta = BTreeMap::new();
        meta.insert("pie_convert".into(), "normalize".into());

        write_zt(
            &path,
            &meta,
            &[
                WriteTensor {
                    decl: &da,
                    bytes: &a,
                },
                WriteTensor {
                    decl: &db,
                    bytes: &b,
                },
            ],
        )
        .unwrap();

        let read = parse_checkpoint(&path).unwrap();
        assert_eq!(read.tensors.len(), 2);
        let ta = read.tensor_by_name("a.weight").unwrap();
        assert_eq!(ta.shape, vec![8, 4]);
        assert_eq!(ta.encoding, Encoding::Raw(DType::F32));
        assert_eq!(ta.span_bytes, a.len() as u64);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The property safetensors cannot give: an MXFP4 payload written and
    /// read back as MXFP4, with its group size intact.
    #[test]
    fn a_quantized_payload_keeps_its_scheme() {
        let dir = tmpdir("quant");
        let path = dir.join("model.zt");
        // 64 logical elements of f4_e2m1 = 32 bytes.
        let payload = vec![0xabu8; 32];
        let spec = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::BF16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: None,
        };
        let d = decl("w", vec![2, 32], Encoding::Quant(spec));
        write_zt(
            &path,
            &BTreeMap::new(),
            &[WriteTensor {
                decl: &d,
                bytes: &payload,
            }],
        )
        .unwrap();

        let read = parse_checkpoint(&path).unwrap();
        let w = read.tensor_by_name("w").unwrap();
        match &w.encoding {
            Encoding::Quant(got) => {
                assert_eq!(got.scheme, QuantScheme::Mxfp4E2M1E8M0);
                assert_eq!(got.group_size, 32);
            }
            other => panic!("expected a quantized encoding, got {other:?}"),
        }
        assert_eq!(w.shape, vec![2, 32]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Streaming a payload in uneven chunks produces the same file as handing
    /// it over whole — how the producer sliced its reads is not the file's
    /// business.
    #[test]
    fn chunked_streaming_matches_whole_bytes() {
        let dir = tmpdir("chunked");
        let bytes: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let d = decl("w", vec![4096], Encoding::Raw(DType::U8));

        let whole = dir.join("whole.zt");
        write_zt(
            &whole,
            &BTreeMap::new(),
            &[WriteTensor {
                decl: &d,
                bytes: &bytes,
            }],
        )
        .unwrap();

        let chunked = dir.join("chunked.zt");
        let mut writer = CheckpointWriter::create(&chunked, &BTreeMap::new()).unwrap();
        writer.begin_tensor(&d, bytes.len() as u64).unwrap();
        for chunk in bytes.chunks(97) {
            writer.write(chunk).unwrap();
        }
        writer.end_tensor().unwrap();
        writer.finish().unwrap();

        assert_eq!(
            std::fs::read(&whole).unwrap(),
            std::fs::read(&chunked).unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two writes of the same tensors produce the same file, byte for byte —
    /// which is what lets an artifact be addressed by its hash.
    #[test]
    fn the_output_is_byte_reproducible() {
        let dir = tmpdir("repro");
        let bytes: Vec<u8> = (0..256).map(|i| (i % 251) as u8).collect();
        let d = decl("w", vec![256], Encoding::Raw(DType::U8));
        let write_to = |name: &str| {
            let path = dir.join(name);
            write_zt(
                &path,
                &BTreeMap::new(),
                &[WriteTensor {
                    decl: &d,
                    bytes: &bytes,
                }],
            )
            .unwrap();
            std::fs::read(&path).unwrap()
        };
        assert_eq!(write_to("a.zt"), write_to("b.zt"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
