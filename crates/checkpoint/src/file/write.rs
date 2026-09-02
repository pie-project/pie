//! Writing a checkpoint: the output side of `convert`. The one writer the
//! loader has; `.zt` is the one format it writes, since its layout lets any
//! scheme the loader can describe be written down (unlike safetensors' dtype
//! tag over a shape).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ztensor::DType as ZDType;
use ztensor::format::cbor::{self, Value};

use crate::error::Error;
use crate::types::{DType, Encoding, QuantScheme, TensorDecl};
use crate::serving::{self, Stamp};

/// One tensor of the file: what to call it, and the bytes as stored.
pub struct WriteTensor<'a> {
    pub decl: &'a TensorDecl,
    pub bytes: &'a [u8],
}

/// The storage type and logical type a declaration's elements are stored as.
///
/// A quantized payload is bytes: the scheme says how to read them, and the
/// layout on the object says which scheme. MXFP4's payload elements are half
/// a byte — `f4_e2m1` says so, and the reader's size check follows from it.
///
/// FP8 must agree with [`profile_of`], which writes it under `dense` plus a
/// logical type. Answering `U8` with no logical type here loses that type on
/// re-read (the reader hands the weight back as `Raw(U8)`).
pub(crate) fn storage_of(
    decl_dtype: DType,
    encoding: &Encoding,
) -> Result<(ZDType, Option<&'static str>), Error> {
    if let Encoding::Quant(spec) = encoding {
        match spec.scheme {
            // Stored as plain elements under `dense`: `decl_dtype` is already
            // the spec's logical type (the F8 width itself).
            QuantScheme::Fp8E4M3 | QuantScheme::Fp8E5M2 => {}
            QuantScheme::Mxfp4E2M1E8M0 => return Ok((ZDType::U8, Some("f4_e2m1"))),
            _ => return Ok((ZDType::U8, None)),
        }
    }
    Ok(match decl_dtype {
        DType::F32 => (ZDType::F32, None),
        DType::F16 => (ZDType::F16, None),
        DType::Bf16 => (ZDType::BF16, None),
        DType::E4m3 => (ZDType::U8, Some("f8_e4m3fn")),
        DType::E5m2 => (ZDType::U8, Some("f8_e5m2")),
        DType::E8m0 => (ZDType::U8, Some("f8_e8m0")),
        DType::I64 => (ZDType::I64, None),
        DType::I32 => (ZDType::I32, None),
        DType::I16 => (ZDType::I16, None),
        DType::I8 => (ZDType::I8, None),
        DType::U64 => (ZDType::U64, None),
        DType::U32 => (ZDType::U32, None),
        DType::U16 => (ZDType::U16, None),
        DType::U8 => (ZDType::U8, None),
        DType::Bool => (ZDType::U8, Some("bool")),
        // Sub-byte codes reach a checkpoint as `U8` under a quant spec (the
        // `Encoding::Quant` arm above); a raw declaration of one has no
        // zTensor dtype.
        DType::E2m1
        | DType::Mxfp4
        | DType::U4g64
        | DType::U8g64
        | DType::U4g32
        | DType::U4g64tiled
        | DType::U2g32
        | DType::U2g64
        | DType::Nvfp4
        | DType::U2g16k
        | DType::I3g16k
        | DType::U4g32k
        | DType::U5g32k
        | DType::I6g16k
        | DType::E4m3row
        | DType::E4m3tile128
        | DType::U2g128 => {
            return Err(Error::Checkpoint(format!(
                "cannot store a raw {decl_dtype:?} tensor: the packed codes \
                 are written as packed U8 under a quant encoding"
            )));
        }
    })
}

/// Everything an object says about its own encoding: the layout profile it is
/// written under, and the attributes a reader recovers it from.
///
/// Two halves answering different questions: [`layout_of`] says how the
/// payload is addressed (what a reader needs to find a block); [`stamp_qnf`]
/// adds what the bytes mean (what a kernel table needs to serve them).
pub(crate) fn profile_of(encoding: &Encoding) -> Result<(&'static str, Option<Value>), Error> {
    let (name, attributes) = layout_of(encoding)?;
    Ok((name, stamp_qnf(attributes, encoding)))
}

/// The `qnf` attribute: what this tensor's bytes mean, as one word.
///
/// The layout profile says how the payload is addressed; two files with the
/// same profile can still hold different arithmetic (AWQ and GPTQ share a
/// row). QNF is the other half: `QuantSpec::term`/`Dtype::repr` map a scheme
/// or dtype onto a term whose mangled spelling is a name a kernel table can
/// key on directly.
///
/// Additive: zTensor's attribute map has no key registry, and this crate's
/// reader looks keys up by name, so a reader built before this attribute
/// existed reads a file carrying it exactly as it read one without.
///
/// `None` from the bridge means no attribute, never a guess: an IQ lattice's
/// points are compiled into llama.cpp rather than stored, so nothing
/// describes their arithmetic here.
fn stamp_qnf(attributes: Option<Value>, encoding: &Encoding) -> Option<Value> {
    let sig = match encoding {
        Encoding::Quant(spec) => spec.term(),
        Encoding::Raw(dtype) => Some(*dtype.repr()),
    };
    // A refusal leaves the profile's own attributes exactly as they were:
    // this function only adds a key or adds nothing.
    let Some(sig) = sig else {
        return attributes;
    };
    // `mangle` panics on a non-canonical term (a factor slot with no
    // factor); nothing in the bridge builds one today.
    let qnf = (
        Value::Text("qnf".to_string()),
        Value::Text(sig.mangle().as_str().to_string()),
    );
    Some(match attributes {
        // The encoder sorts a map by encoded key bytes, so appending is
        // placement-free and the artifact stays byte-reproducible.
        Some(Value::Map(mut entries)) => {
            entries.push(qnf);
            Value::Map(entries)
        }
        // Only reachable if a profile ever returns a non-map value, which
        // the format forbids.
        Some(other) => other,
        None => Value::Map(vec![qnf]),
    })
}

/// The layout profile a declaration's encoding names, and the attributes
/// that make it readable again.
///
/// `zt.quant_group/1` is parametric: what distinguishes AWQ from GPTQ from
/// MLX-affine is the packing order, the scale form and the zero-point form,
/// so those are written down and the scheme's name is not — a reader
/// recovers the scheme from the parameters alone.
fn layout_of(encoding: &Encoding) -> Result<(&'static str, Option<Value>), Error> {
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
            QuantScheme::GgufQ2K => "gguf.q2_k/1",
            QuantScheme::GgufQ3K => "gguf.q3_k/1",
            QuantScheme::GgufQ4_0 => "gguf.q4_0/1",
            QuantScheme::GgufQ4_1 => "gguf.q4_1/1",
            QuantScheme::GgufQ4K => "gguf.q4_k/1",
            QuantScheme::GgufQ5_0 => "gguf.q5_0/1",
            QuantScheme::GgufQ5_1 => "gguf.q5_1/1",
            QuantScheme::GgufQ5K => "gguf.q5_k/1",
            QuantScheme::GgufIq4Nl => "gguf.iq4_nl/1",
            QuantScheme::GgufIq4Xs => "gguf.iq4_xs/1",
            QuantScheme::GgufMxfp4 => "gguf.mxfp4/1",
            QuantScheme::GgufIq2Xxs => "gguf.iq2_xxs/1",
            QuantScheme::GgufIq2Xs => "gguf.iq2_xs/1",
            QuantScheme::GgufIq2S => "gguf.iq2_s/1",
            QuantScheme::GgufIq3Xxs => "gguf.iq3_xxs/1",
            QuantScheme::GgufIq3S => "gguf.iq3_s/1",
            QuantScheme::GgufQ6K => "gguf.q6_k/1",
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
/// The streaming face of [`write_zt`], for callers whose payloads should
/// never all be resident at once. Canonical form requires objects in
/// ascending name order; [`write_zt`] sorts for its caller, this type trusts
/// its caller to add in order.
pub struct Writer {
    /// `None` only after [`finish`](Writer::finish) has taken it.
    writer: Option<ztensor::Writer>,
    open: Option<ztensor::Sink>,
    /// Present when the output is a shard set rather than one file.
    sharding: Option<Sharding>,
    /// The file-level provenance, kept because the serving block below is
    /// merged with it at [`finish`](Writer::finish) and `set_attributes`
    /// replaces rather than merges.
    metadata: BTreeMap<String, String>,
    /// Present when this file is to be a SERVING artifact as well as a
    /// checkpoint — see [`Writer::serving`].
    serving: Option<Serving>,
}

/// What a [`Writer`] accumulates on the way to a `pie.serving/1` file key.
///
/// The tables cannot be written when their objects are declared: an object's
/// attributes are frozen at declaration and a table is a fold over bytes that
/// have not arrived yet. So they are folded here as the payload goes past and
/// handed over at `finish`, which is where the manifest is written. See also
/// `serving::BLOCKS_KEY`.
struct Serving {
    stamp: Stamp,
    /// The open tensor's name and running fold, when the open tensor is one
    /// that will be served. A `__meta__/` object gets neither.
    open: Option<(String, serving::BlockFold)>,
    tables: BTreeMap<String, BTreeMap<String, Vec<u8>>>,
}

/// The state of a multi-file write: where the root goes, which shard is open,
/// and what the root will have to say about them.
///
/// A shard is an ordinary `.zt`, not a bare blob heap: zTensor's bare kind
/// takes each blob as one `&[u8]`, which would mean holding a whole tensor in
/// memory, so shards are written through the same streaming path single-file
/// artifacts use, and the root references their bytes with [`Writer::link`]
/// (copies nothing, carries each part's digest across). Every shard verifies
/// and opens on its own, and the root stays small enough to be cheap to read.
struct Sharding {
    root: PathBuf,
    attributes: BTreeMap<String, String>,
    /// Soft cap. A tensor is never split, so a shard closes after the tensor
    /// that crossed the line, and one tensor larger than the cap gets a shard
    /// to itself rather than an error.
    max_bytes: u64,
    /// Bytes declared into the shard currently open.
    current_bytes: u64,
    /// 1-based index of the shard currently open.
    index: u32,
    /// Shards already published, in order: their table names and paths.
    done: Vec<(String, PathBuf)>,
    /// Metadata objects, held back for the root: opening one file has to
    /// answer "what model is this" regardless of which shard you open.
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
/// Five digits: the table holds `00001` and the resolver looks beside the
/// root for `<stem>-00001.zt`. The table never holds a path — a name is all
/// a shard reference carries, since a file name is not something the format
/// can verify.
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

impl Writer {
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
        Self::opened(path, metadata, None)
    }

    /// [`create`](Self::create), for a file that is to be a serving artifact
    /// as well as a checkpoint.
    ///
    /// It adds exactly one file attribute, keyed `pie.serving/1`, holding the
    /// stamp's members and every served object's block table, folded from the
    /// payload as it streams past. Delete that key and an ordinary checkpoint
    /// of the same weights remains.
    ///
    /// A constructor, not a setter, because placement matters: a serving
    /// artifact's objects are written in the boot's read order, not name
    /// order, and `ztensor`'s canonical form requires ascending insertion and
    /// refuses anything else — so the non-canonical label has to be set
    /// before the first object is written. Only the label is given up; a
    /// non-canonical writer still places on 64 KiB and no reader checks the
    /// label. The order itself is the caller's — this does not re-sort it; a
    /// caller that adds out of order writes a file that reads unranked
    /// (`serving::sequence`'s own admitted limit).
    pub fn create_serving(
        path: &Path,
        metadata: &BTreeMap<String, String>,
        stamp: Stamp,
    ) -> Result<Self, Error> {
        Self::opened(path, metadata, Some(stamp))
    }

    fn opened(
        path: &Path,
        metadata: &BTreeMap<String, String>,
        stamp: Option<Stamp>,
    ) -> Result<Self, Error> {
        let mut writer = match &stamp {
            // Canonical form requires ascending insertion and a serving
            // artifact's order is the boot's — see `create_serving`.
            Some(_) => ztensor::Writer::options()
                .canonical(false)
                .publish(path)
                .map_err(Error::from)?,
            None => ztensor::Writer::publish(path).map_err(Error::from)?,
        };
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
            metadata: metadata.clone(),
            serving: stamp.map(|stamp| Serving {
                stamp,
                open: None,
                tables: BTreeMap::new(),
            }),
        })
    }

    /// Opens a checkpoint that spills into shards once one file passes
    /// `max_shard_bytes`.
    ///
    /// The output is a root `.zt` beside `<stem>-00001.zt`, `<stem>-00002.zt`,
    /// … — zTensor's native multi-file model. The root carries the manifest,
    /// the metadata and a shard table naming each shard by size and digest;
    /// the shards carry the weights.
    ///
    /// The cap is soft: a tensor is never split across shards, so a shard
    /// closes after whichever tensor crossed the line, and a tensor bigger
    /// than the whole cap gets a shard to itself.
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
            metadata: metadata.clone(),
            serving: None,
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
        crate::file::meta::reject_reserved(&decl.name)?;
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
        if let Some(serving) = &mut self.serving {
            serving.open = serving::is_serving(&decl.name).then(|| {
                (
                    decl.name.clone(),
                    serving::BlockFold::new(
                        serving.stamp.block_algorithm,
                        serving.stamp.block_bytes,
                    ),
                )
            });
        }
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
        // The fold takes the same slice the container is handed, on the way
        // in, so the table describes what was written rather than what was
        // meant.
        if let Some((_, fold)) = self.serving.as_mut().and_then(|it| it.open.as_mut()) {
            fold.eat(chunk);
        }
        sink.write(writer, chunk).map_err(Error::from)
    }

    /// Closes the open tensor, which must have received its whole payload.
    pub fn end_tensor(&mut self) -> Result<(), Error> {
        let sink = self
            .open
            .take()
            .ok_or_else(|| Error::Checkpoint("no tensor is open".into()))?;
        if let Some(serving) = self.serving.as_mut()
            && let Some((name, fold)) = serving.open.take()
        {
            // Single-part by construction: `begin_tensor` declares one part
            // and names it `data`, so a checkpoint's streamed object is
            // always a one-part object and the table map has one entry.
            serving
                .tables
                .insert(name, BTreeMap::from([("data".to_string(), fold.finish())]));
        }
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
        let name = crate::file::meta::meta_name(path);
        write_meta_object(self.writer(), &name, bytes)
    }

    /// Closes the open shard and opens the next, when the one open is full.
    ///
    /// Called before a tensor is declared, so the decision is "does this
    /// tensor still fit" rather than "did the last one overflow" — which
    /// keeps a tensor whole. An empty shard never rolls, so a tensor larger
    /// than the cap lands in one of its own instead of an empty file.
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
    /// Non-canonical, because canonical form is single-file by definition.
    /// What that costs is only the label: a non-canonical writer still
    /// places on 64 KiB, and nothing on disk records the label or checks it.
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
        // The serving key goes on the root, the only manifest there is: the
        // shards carry blobs, the root carries every object's description.
        // Nothing constructs a sharded serving writer yet (`create_serving`
        // is single-file, `create_sharded` takes no stamp), so this arm is
        // written and waiting rather than running.
        if let Some(serving) = self.serving.take() {
            root.set_attributes(serving_attributes(&serving, &sharding.attributes)?);
        } else if !sharding.attributes.is_empty() {
            root.set_attributes(Value::Map(
                sharding
                    .attributes
                    .iter()
                    .map(|(k, v)| (Value::Text(k.clone()), Value::Text(v.clone())))
                    .collect(),
            ));
        }
        for (name, path) in &shards {
            // The identity is read back from the finished shard rather than
            // accumulated while writing: a digest of the bytes actually
            // there. XXH3 (not SHA-256) since a local-store artifact is
            // neither signed nor distributed, and it's the digest every
            // tensor in it already carries.
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
            let name = crate::file::meta::meta_name(path);
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
        let serving = self.serving.take();
        let metadata = std::mem::take(&mut self.metadata);
        let mut writer = self.writer.take().expect("writer present");
        if let Some(serving) = serving {
            writer.set_attributes(serving_attributes(&serving, &metadata)?);
        }
        writer.finish().map_err(Error::from)?;
        Ok(())
    }

    fn writer(&mut self) -> &mut ztensor::Writer {
        self.writer.as_mut().expect("writer present")
    }
}

/// The file attributes of a serving artifact: the profile's one key, and the
/// flat provenance beside it.
///
/// A provenance key that collides with the profile's is refused rather than
/// resolved: a caller passing one has two beliefs about the serving block,
/// and either choice makes the file say something nobody wrote.
fn serving_attributes(
    serving: &Serving,
    metadata: &BTreeMap<String, String>,
) -> Result<Value, Error> {
    let Value::Map(mut entries) = serving::file_block(&serving.stamp, &serving.tables) else {
        return Err(Error::Internal(
            "the stamp encoded to something that is not a map".to_string(),
        ));
    };
    for (key, value) in metadata {
        if key.starts_with(serving::PROFILE_FAMILY) {
            return Err(Error::Checkpoint(format!(
                "the provenance key {key:?} is the key the stamp itself is written \
                 under, so this artifact would carry two answers for it"
            )));
        }
        entries.push((Value::Text(key.clone()), Value::Text(value.clone())));
    }
    Ok(Value::Map(entries))
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
    let mut writer = Writer::create(path, metadata)?;
    let mut ordered: Vec<&WriteTensor<'_>> = tensors.iter().collect();
    ordered.sort_by(|a, b| a.decl.name.cmp(&b.decl.name));
    for tensor in ordered {
        writer.add_tensor(tensor.decl, tensor.bytes)?;
    }
    writer.finish()
}

