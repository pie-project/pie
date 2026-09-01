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
use crate::serving::{self, Stamp};

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
///
/// FP8 is not an exception so much as not a group code at all, and it is the
/// one case where this function has to agree with [`profile_of`] rather than
/// decide on its own: that function writes FP8 under `dense` because "FP8
/// weights are not group-quantized codes: they are plain f8 elements … `dense`
/// plus the logical type says that exactly". Answering `U8` with no logical
/// type here writes only the first half of that sentence, and the reader —
/// which takes `dense` to mean the parts are plain values — hands the weight
/// back as `Raw(U8)` with its type gone.
///
/// Measured on `pie model build --backend cuda --quant fp8 --fp8-native` from
/// `Qwen--Qwen3-0.6B.zt` (a command R3 deleted -- the finding stands, the
/// recipe cannot be re-run as written), before and after: **196 tensors**
/// carried no logical
/// type, the artifacts differ by 2,940 bytes of metadata and **not one byte of
/// payload**, and the erasure *propagates* — re-authoring a type-erased
/// artifact writes another one, because the reader answers `U8` and the writer
/// stores what it was told.
///
/// Serving was never affected: both artifacts re-author to the same 506
/// tensors. What broke is using a built FP8 artifact as a quantization source,
/// which refused with "runtime_quant source 'model.layers.0.mlp.down_proj.weight'
/// must be BF16/FP16/FP32/F8E4M3" — the weight *was* F8E4M3, and the file no
/// longer said so.
/// `every_scheme_the_writer_accepts_is_one_the_reader_gives_back` is what found
/// this and is what keeps the two functions agreeing.
pub(crate) fn storage_of(
    decl_dtype: DType,
    encoding: &Encoding,
) -> Result<(ZDType, Option<&'static str>), Error> {
    if let Encoding::Quant(spec) = encoding {
        match spec.scheme {
            // Stored as plain elements under `dense`, so the dtype table
            // below is the right answer — `decl_dtype` is already the spec's
            // logical type, which for these schemes is the F8 width itself.
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
        // The two sub-byte codes reach a checkpoint as `U8` under a quant
        // spec, which the `Encoding::Quant` arm above already answered. A
        // *declaration* naming one of them is a plane with no element width
        // to store, and there is no zTensor dtype for it.
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
/// Two halves, and they answer different questions. [`layout_of`] says how the
/// payload is ADDRESSED, which is what a reader needs to find a block; then
/// [`stamp_qnf`] adds what the bytes MEAN, which is what a kernel table needs
/// to decide it can serve them.
pub(crate) fn profile_of(encoding: &Encoding) -> Result<(&'static str, Option<Value>), Error> {
    let (name, attributes) = layout_of(encoding)?;
    Ok((name, stamp_qnf(attributes, encoding)))
}

/// The `qnf` attribute: what this tensor's bytes MEAN, as one word.
///
/// The layout profile above says how the payload is *addressed* —
/// `gguf.q4_k/1`, `zt.quant_group/1` plus its parameters, `dense` — and two
/// files with the same profile can still hold different arithmetic (AWQ and
/// GPTQ share a row; `dense` covers every plain width there is). QNF is the
/// other half: `QuantSpec::term` maps a scheme, and `Dtype::repr` a dtype, onto a term, and its
/// mangled spelling is a name a kernel table can be keyed on directly, with no
/// second scheme enum in between.
///
/// **ADDITIVE, AND THE CONTAINER SAYS SO.** zTensor's attribute rule (spec
/// §3.1/§3.5, `format::check_attributes`) is that the value is a map whose
/// top-level keys are text obeying the name rules — no key registry, no
/// closed-world check — and this crate's own reader looks its keys up by name
/// (`file/zt.rs::encoding_of` asks for `group_size`, `bits`, `axis`,
/// `elems_per_block`) and never enumerates them. So a reader built before this
/// attribute existed reads a file carrying it exactly as it read one without.
///
/// **`None` FROM THE BRIDGE MEANS NO ATTRIBUTE, NEVER A GUESS.** That is
/// `crate::qnf`'s own rule and the reason it answers `Option`: an IQ lattice's
/// points are compiled into llama.cpp rather than stored, so no group width and
/// no code leaf describes what its bytes hold. Such a tensor is written with
/// its profile and without this attribute, which is the honest statement that
/// the tree cannot yet name its arithmetic.
fn stamp_qnf(attributes: Option<Value>, encoding: &Encoding) -> Option<Value> {
    let sig = match encoding {
        Encoding::Quant(spec) => spec.term(),
        Encoding::Raw(dtype) => Some(*dtype.repr()),
    };
    // A refusal leaves the profile's own attributes exactly as they were: this
    // function ADDS a key or adds nothing, and dropping `bits`/`group_size`
    // here would write a file the reader cannot open — which is what the first
    // draft of it did, for every scheme the bridge has no row for.
    //
    // `is_canonical` is asked because `mangle` PANICS otherwise: a factor slot
    // with no factor has no spelling. Nothing in the bridge builds such a term
    // today, and this is what keeps that a fact rather than a hope.
    let Some(sig) = sig else {
        return attributes;
    };
    let qnf = (
        Value::Text("qnf".to_string()),
        Value::Text(sig.mangle().as_str().to_string()),
    );
    Some(match attributes {
        // The encoder sorts a map by encoded key bytes (`format::cbor`), so
        // appending is placement-free: the file is the same whatever order the
        // pairs arrive in, and the artifact stays byte-reproducible.
        Some(Value::Map(mut entries)) => {
            entries.push(qnf);
            Value::Map(entries)
        }
        // Only reachable if a profile ever returns a non-map attribute value,
        // which the format forbids; keeping it rather than replacing it means
        // this never silently drops what a profile said.
        Some(other) => other,
        None => Value::Map(vec![qnf]),
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
/// The streaming face of [`write_zt`], for callers whose payloads should never
/// all be resident at once — `pie model import` copies multi-gigabyte
/// tensors straight from the source checkpoint through a bounded buffer.
/// Canonical form requires objects in ascending name order; [`write_zt`] sorts
/// for its caller, this type trusts its caller to add in order.
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
/// handed over at `finish`, which is where the manifest is written. The same
/// argument, and the 95.4 GiB plane that forced it, is at
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

    /// [`create`](Self::create), for a file that is to be a SERVING artifact
    /// as well as a checkpoint — §M-4's one `.zt`, which is both.
    ///
    /// It adds exactly one file attribute, keyed `pie.serving/1`, holding the
    /// stamp's members and every served object's block table, folded from the
    /// payload as it streams past. Delete that key and an ordinary checkpoint
    /// of the same weights remains, which is the owner's rule and the reason
    /// the vocabulary is one key in one place.
    ///
    /// # It is a CONSTRUCTOR and not a setter, and the reason is placement
    ///
    /// A serving artifact's objects are written in the BOOT'S READ ORDER,
    /// which is not name order — and `ztensor`'s canonical form requires
    /// ascending insertion and refuses anything else outright
    /// (`"canonical form requires sorted insertion: \"embed\" after
    /// \"layer.0.qg_proj\""`). So the label has to be off from the first
    /// object, which is before any method could be called. That the ordinary
    /// [`create`](Self::create) stays canonical is deliberate: nothing else
    /// in this tree wants a non-ascending checkpoint, and the label is cheap
    /// to keep where it is true.
    ///
    /// The label is all that is given up. `ztensor::write::Options::canonical`
    /// says it from the other side — *"Placement is not part of what you give
    /// up"* — and since 2.1.0 a non-canonical writer still places on 64 KiB.
    /// Nothing on disk records the label and no reader checks it.
    ///
    /// **The ORDER itself is the caller's and this does not touch it.** A
    /// writer that re-sorted would be overruling the only party that knows
    /// the read order; a caller that adds in some other order writes a file
    /// this build reads perfectly and merely reads unranked, which is
    /// `serving::sequence`'s own admitted limit.
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
        // The fold takes the SAME slice the container is handed, on the way
        // in, so the table describes what was written rather than what was
        // meant. Splitting the two would be the one bug this attribute cannot
        // survive: a wrong table refuses a good file at every boot.
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
        // **THE SERVING KEY GOES ON THE ROOT**, which is the only manifest
        // there is: the shards carry blobs and the root carries every
        // object's description, so a table keyed by object name addresses a
        // part in a shard exactly as the manifest's own blob reference does.
        //
        // **AND NOTHING CONSTRUCTS A SHARDED SERVING WRITER YET.**
        // `create_serving` is single-file and `create_sharded` takes no
        // stamp, so this arm is written and waiting rather than running. It
        // is here rather than as a refusal because the fact it states — the
        // root is where the key goes — is the answer whenever the import
        // starts sharding a serving artifact, and a refusal would have to be
        // deleted to be replaced by exactly these four lines.
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
/// A provenance key that collides with the profile's is REFUSED rather than
/// resolved, for `file/emit.rs`'s reason and in its words: a caller passing
/// one has two beliefs about the serving block, and either choice makes the
/// file say something nobody wrote.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::zt::parse;
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
        use crate::file::meta;
        use crate::contract::infer::CheckpointTypes;

        let dir = tmpdir("meta");
        let path = dir.join("model.zt");
        let weight: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let descriptor = br#"{"arch":"llama3","hidden_size":64}"#;
        let vocab = b"<s></s>hello world";

        let mut writer = Writer::create(&path, &BTreeMap::new()).unwrap();
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

        let parsed = parse(&path).unwrap();

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
        let sources = crate::file::Sources::new(&parsed);
        assert!(sources.by_name("model.embed.weight").is_some());
        assert!(sources.by_name("__meta__/tokenizer/vocab_bytes").is_none());
        assert!(
            parsed
                .tensor_type("__meta__/tokenizer/vocab_bytes")
                .is_none()
        );

        // And a weight cannot be written into the namespace at all.
        let mut writer = Writer::create(&dir.join("bad.zt"), &BTreeMap::new()).unwrap();
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
        use crate::file::meta;

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
                Writer::create_sharded(path, &BTreeMap::new(), 100_000).unwrap()
            } else {
                Writer::create(path, &BTreeMap::new()).unwrap()
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
        let single = parse(&one).unwrap();
        let sharded = parse(&many).unwrap();
        let names = |cp: &crate::file::Metadata| -> Vec<String> {
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
            crate::file::zt::verify(&many).unwrap(),
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
    /// exclusivity is what lets the engine mmap-stream routed experts), and a
    /// regression in it would be invisible: every file still reads back fine.
    /// So it is checked here rather than assumed.
    #[test]
    fn a_shard_set_places_on_64_kib() {
        let dir = tmpdir("shard-align");
        let root = dir.join("model.zt");
        let mut writer = Writer::create_sharded(&root, &BTreeMap::new(), 100_000).unwrap();
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

        let mut writer = Writer::create_sharded(&root, &BTreeMap::new(), 8_000).unwrap();
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

        let parsed = parse(&root).unwrap();
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

        assert_eq!(crate::file::zt::read_attributes(&path).unwrap(), attributes);
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

        let read = parse(&path).unwrap();
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
            logical_dtype: DType::Bf16,
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

        let read = parse(&path).unwrap();
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
        let mut writer = Writer::create(&chunked, &BTreeMap::new()).unwrap();
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

    /// The two GGUF tables are separate halves of one fact, and nothing but
    /// this holds them level.
    ///
    /// A scheme that reports a `block_layout` is a scheme whose bytes this
    /// writer will be asked to carry, so a missing profile is not a gap that
    /// shows up in review -- it shows up as a refusal partway through writing
    /// a 12 GB artifact, which is where `GgufQ6K` was found after its reader
    /// half had already been added.
    #[test]
    fn every_blocked_scheme_can_be_written_back() {
        for scheme in [
            QuantScheme::GgufQ4_0,
            QuantScheme::GgufQ4K,
            QuantScheme::GgufQ5_0,
            QuantScheme::GgufQ5K,
            QuantScheme::GgufQ6K,
            QuantScheme::GgufQ8_0,
        ] {
            let spec = QuantSpec {
                scheme,
                logical_dtype: DType::Bf16,
                bits_per_element: 0,
                group_size: 0,
                channel_axis: None,
            };
            assert!(
                spec.scheme.is_self_contained(),
                "{scheme:?} is a blocked scheme"
            );
            let (name, _) =
                profile_of(&Encoding::Quant(spec)).expect("a profile to write it under");
            assert!(
                name.starts_with("gguf."),
                "{scheme:?} lands on {name}, which is not a gguf profile"
            );
        }
    }

    /// What the reader gives back for a scheme the writer was handed.
    ///
    /// Not every scheme comes back as itself, and the ones that do not are a
    /// decision rather than a gap: FP8 weights are plain elements, not group
    /// codes, so they are stored under `dense` and the reader answers with a
    /// dtype and no scheme.
    #[derive(Debug, PartialEq, Eq)]
    enum RoundTrip {
        /// Written under a quantization profile; read back as the same scheme.
        Survives,
        /// Written as `dense`; read back as a plain dtype, scheme discarded.
        Collapses(DType),
        /// Not a spelling the rest of the loader builds. `Encoding::Raw` is
        /// how an unquantized tensor is written, so `Quant { scheme: None }`
        /// has no round trip to check.
        NotAnEncoding,
    }

    /// The round trip each scheme must make, stated once.
    ///
    /// **Exhaustive on purpose, and it is the only completeness check either
    /// half of the round trip has.** `profile_of` ends in `other => Err(...)`
    /// and `scheme_of` matches strings, so neither is checked by the
    /// compiler: a new `QuantScheme` compiles clean and fails at runtime, on
    /// the first artifact somebody writes with it. `rustc` refuses this match
    /// until a new variant is given an answer, which is what makes forgetting
    /// the list below expensive rather than silent.
    fn round_trip_of(scheme: QuantScheme) -> RoundTrip {
        match scheme {
            QuantScheme::None => RoundTrip::NotAnEncoding,
            QuantScheme::Fp8E4M3 => RoundTrip::Collapses(DType::E4m3),
            QuantScheme::Fp8E5M2 => RoundTrip::Collapses(DType::E5m2),
            QuantScheme::Int8Symmetric
            | QuantScheme::Int8Asymmetric
            | QuantScheme::AwqInt4
            | QuantScheme::GptqInt4
            | QuantScheme::Mxfp4E2M1E8M0
            | QuantScheme::MlxAffineU4
            | QuantScheme::Int4B8
            | QuantScheme::GgufQ4_0
            | QuantScheme::GgufQ4_1
            | QuantScheme::GgufQ2K
            | QuantScheme::GgufQ3K
            | QuantScheme::GgufQ4K
            | QuantScheme::GgufQ5_0
            | QuantScheme::GgufQ5_1
            | QuantScheme::GgufQ5K
            | QuantScheme::GgufQ6K
            | QuantScheme::GgufQ8_0
            | QuantScheme::GgufIq4Nl
            | QuantScheme::GgufIq4Xs
            | QuantScheme::GgufMxfp4
            | QuantScheme::GgufIq2Xxs
            | QuantScheme::GgufIq2Xs
            | QuantScheme::GgufIq2S
            | QuantScheme::GgufIq3Xxs
            | QuantScheme::GgufIq3S => RoundTrip::Survives,
        }
    }

    /// Every variant, in declaration order.
    ///
    /// Hand-maintained, and kept honest the way
    /// `manifest::from_checkpoint::tests::no_scheme_has_a_zero_width_so_no_division_can_fault`
    /// keeps its own copy honest: [`round_trip_of`] is exhaustive so a new
    /// variant cannot compile without an answer, and the length assertion in
    /// the test below fails if the answer was written without adding the
    /// variant here.
    const EVERY_SCHEME: &[QuantScheme] = &[
        QuantScheme::None,
        QuantScheme::Fp8E4M3,
        QuantScheme::Fp8E5M2,
        QuantScheme::Int8Symmetric,
        QuantScheme::Int8Asymmetric,
        QuantScheme::AwqInt4,
        QuantScheme::GptqInt4,
        QuantScheme::Mxfp4E2M1E8M0,
        QuantScheme::MlxAffineU4,
        QuantScheme::GgufQ4_0,
        QuantScheme::GgufQ2K,
        QuantScheme::GgufQ3K,
        QuantScheme::GgufQ4K,
        QuantScheme::GgufQ5_0,
        QuantScheme::GgufQ5K,
        QuantScheme::GgufQ8_0,
        QuantScheme::Int4B8,
        QuantScheme::GgufQ6K,
        QuantScheme::GgufQ4_1,
        QuantScheme::GgufQ5_1,
        QuantScheme::GgufIq4Nl,
        QuantScheme::GgufIq4Xs,
        QuantScheme::GgufMxfp4,
        QuantScheme::GgufIq2Xxs,
        QuantScheme::GgufIq2Xs,
        QuantScheme::GgufIq2S,
        QuantScheme::GgufIq3Xxs,
        QuantScheme::GgufIq3S,
    ];

    /// The writer's profile is readable by the reader that has to read it,
    /// for every scheme, through a real file.
    ///
    /// `profile_of` writes twelve layout profiles and
    /// [`scheme_of`](crate::file::zt) reads twelve back, in two
    /// modules, with no shared table and nothing checking they agree. They
    /// have to: an artifact whose layout the reader cannot resolve is refused
    /// at parse, so a scheme that writes under a profile nobody reads
    /// produces a file that pie itself cannot open.
    ///
    /// End to end rather than `scheme_of(profile_of(s))` because the seam is
    /// wider than those two functions: the attributes go out as CBOR and come
    /// back parsed, and `zt.quant_group/1` does not carry a scheme name at
    /// all — the reader *derives* it from `bits`, `packing`, `scale_form` and
    /// `zero_point`, so six schemes share one profile and are told apart only
    /// by values that survive a serialization round trip.
    /// **THE TWO WRITERS AGREE, TABLE FOR TABLE.**
    ///
    /// `file/emit.rs` writes a serving artifact from objects handed to it;
    /// this `Writer` writes one as `pie model import` streams a checkpoint
    /// through it. Both produce the same file attribute, and nothing in the
    /// type system says they must — which is exactly the shape
    /// `every_scheme_the_writer_accepts_is_one_the_reader_gives_back` exists
    /// for one level down, and the reason `serving::BlockFold` is one value
    /// both call rather than a loop each spells.
    ///
    /// A wrong table is the one thing this attribute cannot survive: it
    /// refuses a GOOD file at every boot, and the payload is fine, so nothing
    /// downstream is in a position to notice which half is lying.
    ///
    /// The planes are added in a NON-name order on purpose. Both writers must
    /// leave the order alone — a serving artifact's sequence is the boot's
    /// read order and neither of them knows it — so a writer that sorted
    /// would be caught here rather than by a cold first light.
    #[test]
    fn the_streaming_writer_states_what_the_emitter_states() {
        use crate::file::emit::{self, Object, Payload, Part};
        use crate::serving::{BlockAlgorithm, Stamp};

        let dir = tmpdir("two-writers");
        let stamp = Stamp {
            serving: crate::serving::PROFILE.to_string(),
            backend: "cuda".to_string(),
            tp_size: 1,
            sku: "qwen_3".to_string(),
            precision: "bf16".to_string(),
            layout_revision: 1,
            block_bytes: crate::serving::MIN_BLOCK_BYTES,
            block_algorithm: BlockAlgorithm::Xxh3,
            adapters_zeroed: true,
            model_id: None,
            recipe_digest: None,
        };
        let block = crate::serving::MIN_BLOCK_BYTES as usize;
        let planes: Vec<(&str, Vec<u8>)> = vec![
            ("layer.0.qg_proj", vec![0x11u8; 3 * block + 7]),
            ("embed", vec![0x22u8; block - 1]),
            ("layer.0.norm", vec![0x33u8; 2 * block]),
        ];

        // The emitter, bytes in hand.
        let emitted = dir.join("emitted.zt");
        let objects: Vec<Object<'_>> = planes
            .iter()
            .map(|(name, bytes)| Object {
                name,
                shape: vec![bytes.len() as u64],
                layout: "dense",
                attributes: None,
                parts: vec![Part {
                    name: "data",
                    dtype: ZDType::U8,
                    logical: None,
                    payload: Payload::Whole(bytes),
                }],
            })
            .collect();
        emit::write(
            &emitted,
            &stamp,
            &BTreeMap::new(),
            crate::serving::MIN_BLOCK_BYTES,
            &objects,
            |object, part, _| panic!("{object}/{part} is not streamed here"),
        )
        .unwrap();

        // The streaming writer, in 97-byte pieces so no chunk is a block.
        let streamed = dir.join("streamed.zt");
        let mut writer =
            Writer::create_serving(&streamed, &BTreeMap::new(), stamp.clone()).unwrap();

        for (name, bytes) in &planes {
            let d = decl(name, vec![bytes.len() as i64], Encoding::Raw(DType::U8));
            writer.begin_tensor(&d, bytes.len() as u64).unwrap();
            for piece in bytes.chunks(97) {
                writer.write(piece).unwrap();
            }
            writer.end_tensor().unwrap();
        }
        writer.finish().unwrap();

        let table_of = |path: &std::path::Path, object: &str| -> Vec<u8> {
            let manifest = ztensor::read::manifest_of(path).unwrap().unwrap();
            crate::serving::stated_blocks(manifest.attributes.as_ref().unwrap(), object, "data")
                .unwrap_or_else(|| panic!("{} states no table for {object}", path.display()))
                .to_vec()
        };
        for (name, bytes) in &planes {
            assert_eq!(
                table_of(&streamed, name),
                table_of(&emitted, name),
                "the two writers disagree about {name}'s block table",
            );
            assert_eq!(
                table_of(&streamed, name).len(),
                crate::serving::table_len(
                    bytes.len() as u64,
                    crate::serving::MIN_BLOCK_BYTES,
                    BlockAlgorithm::Xxh3,
                ),
                "{name}'s table is not as long as its bytes say",
            );
        }
        // And the sequence is the order each was handed, not the names.
        let sequence = |path: &std::path::Path| -> Vec<String> {
            let manifest = ztensor::read::manifest_of(path).unwrap().unwrap();
            crate::serving::sequence(&manifest)
                .into_iter()
                .map(str::to_string)
                .collect()
        };
        let want: Vec<String> = planes.iter().map(|(n, _)| (*n).to_string()).collect();
        assert_eq!(sequence(&streamed), want);
        assert_eq!(sequence(&emitted), want);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn every_scheme_the_writer_accepts_is_one_the_reader_gives_back() {
        use crate::types::Axis;

        let dir = tmpdir("roundtrip");
        for &scheme in EVERY_SCHEME {
            let expected = round_trip_of(scheme);
            if expected == RoundTrip::NotAnEncoding {
                continue;
            }
            let spec = QuantSpec {
                scheme,
                logical_dtype: match scheme {
                    QuantScheme::Fp8E4M3 => DType::E4m3,
                    QuantScheme::Fp8E5M2 => DType::E5m2,
                    _ => DType::Bf16,
                },
                bits_per_element: 0,
                group_size: 0,
                channel_axis: Some(Axis(0)),
            }
            .normalized();

            // One block for a blocked scheme, one group for the rest. The
            // sizes come from the scheme rather than from a literal so that a
            // scheme whose geometry changes is exercised at its new geometry.
            let (elems, nbytes) = match spec.block_layout() {
                Some((elems, bytes)) => (elems as i64, bytes as usize),
                None => {
                    let group = u64::from(spec.normalized_group_size()).max(1);
                    let bits = u64::from(spec.normalized_bits());
                    (
                        group as i64,
                        usize::try_from((group * bits).div_ceil(8)).unwrap(),
                    )
                }
            };

            let path = dir.join(format!("{scheme:?}.zt"));
            let d = decl("w", vec![1, elems], Encoding::Quant(spec));
            write_zt(
                &path,
                &BTreeMap::new(),
                &[WriteTensor {
                    decl: &d,
                    bytes: &vec![0x5au8; nbytes],
                }],
            )
            .unwrap_or_else(|err| panic!("{scheme:?} could not be written: {err}"));

            let read = parse(&path)
                .unwrap_or_else(|err| panic!("{scheme:?} wrote a file pie cannot open: {err}"));
            let got = &read.tensor_by_name("w").unwrap().encoding;

            match (&expected, got) {
                (RoundTrip::Survives, Encoding::Quant(got)) => assert_eq!(
                    got.scheme, scheme,
                    "{scheme:?} was written and read back as {:?}",
                    got.scheme
                ),
                (RoundTrip::Collapses(dtype), Encoding::Raw(got)) => assert_eq!(
                    got, dtype,
                    "{scheme:?} collapses to a plain dtype, but not to {dtype:?}"
                ),
                _ => panic!("{scheme:?}: expected {expected:?}, read back {got:?}"),
            }
        }
        assert_eq!(
            EVERY_SCHEME.len(),
            28,
            "a scheme was added; give it a case above"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The six schemes that share `zt.quant_group/1`, and the widths each is
    /// written and read at.
    ///
    /// The profile carries no scheme name — the reader derives one from the
    /// stated parameters — so these six are the only rows where two schemes
    /// can be confused for each other, and `bits` is a parameter they share
    /// rather than a name any of them owns.
    const AFFINE_WIDTHS: &[(QuantScheme, &[u8])] = &[
        // Four and only four, and [`QuantSpec::term`] says the same in its
        // own words: `zt.quant_group/1` names these at four bits and there.
        (QuantScheme::AwqInt4, &[4]),
        (QuantScheme::GptqInt4, &[4]),
        (QuantScheme::Int4B8, &[4]),
        // **THE ROW WITH THREE.** Two for the DQ expert banks, four for the
        // ordinary projections, eight for the MoE router gates a
        // `quant_predicate` lifts — one arithmetic with a number in it, which
        // is `Dtype::U8g64`'s and `Dtype::U2g32`'s whole argument.
        (QuantScheme::MlxAffineU4, &[2, 4, 8]),
        (QuantScheme::Int8Symmetric, &[8]),
        (QuantScheme::Int8Asymmetric, &[8]),
    ];

    /// Every width a scheme is written at reads back as THAT scheme, and no
    /// width reads back as a different one.
    ///
    /// [`every_scheme_the_writer_accepts_is_one_the_reader_gives_back`] walks
    /// each scheme once, at the width `normalized` fills in — MLX's default
    /// is four — so the two widths MLX gained after that default was chosen
    /// were never written by any test. Both were broken, in the two different
    /// ways a shared profile can break:
    ///
    /// - at **two**, the reader had no row and refused a file THIS WRITER had
    ///   just produced ("names no scheme this loader implements"), which is
    ///   loud and would have been found the first time a 2-bit artifact was
    ///   opened;
    /// - at **eight**, an MLX bank matched `Int8Asymmetric`'s row — same
    ///   packing order, same zero-point form, same zero-point packing — and
    ///   came back as a scheme whose factors are f32 where MLX's are bf16.
    ///   Silent, and wrong at the kernel.
    ///
    /// So the sweep has two halves. The positive half is [`AFFINE_WIDTHS`]:
    /// a width a scheme is listed at must survive the trip, scheme AND width.
    /// The negative half needs no table and is the one that would have caught
    /// the eight: at every width from one to sixteen, a file that parses at
    /// all must come back as the scheme that wrote it. A REFUSAL is a fine
    /// answer there and a different scheme never is.
    #[test]
    fn a_shared_profile_gives_every_width_back_to_the_scheme_that_wrote_it() {
        use crate::types::Axis;

        let dir = tmpdir("affine-widths");
        let write_at = |scheme: QuantScheme, bits: u8| -> Result<QuantSpec, Error> {
            let spec = QuantSpec {
                scheme,
                logical_dtype: DType::Bf16,
                bits_per_element: bits,
                group_size: 0,
                channel_axis: Some(Axis(0)),
            }
            .normalized();
            let group = u64::from(spec.normalized_group_size()).max(1);
            let nbytes = usize::try_from((group * u64::from(bits)).div_ceil(8)).unwrap();
            let path = dir.join(format!("{scheme:?}-{bits}.zt"));
            let d = decl("w", vec![1, group as i64], Encoding::Quant(spec));
            write_zt(
                &path,
                &BTreeMap::new(),
                &[WriteTensor {
                    decl: &d,
                    bytes: &vec![0x5au8; nbytes],
                }],
            )?;
            let read = parse(&path)?;
            match &read.tensor_by_name("w").unwrap().encoding {
                Encoding::Quant(got) => Ok(got.clone()),
                other => panic!("{scheme:?} at {bits} bits came back {other:?}, not quantized"),
            }
        };

        for &(scheme, widths) in AFFINE_WIDTHS {
            for &bits in widths {
                let got = write_at(scheme, bits).unwrap_or_else(|err| {
                    panic!("{scheme:?} at {bits} bits does not survive its own writer: {err}")
                });
                assert_eq!(
                    got.scheme, scheme,
                    "{scheme:?} at {bits} bits was read back as {:?}",
                    got.scheme
                );
                assert_eq!(
                    got.bits_per_element, bits,
                    "{scheme:?} kept its name at {bits} bits and lost its width"
                );
            }
            // The half that needs no table: a width off the list may be
            // refused, and must never be answered with somebody else's row.
            for bits in 1u8..=16 {
                if let Ok(got) = write_at(scheme, bits) {
                    assert_eq!(
                        got.scheme, scheme,
                        "{scheme:?} written at {bits} bits reads back as {:?} — a shared \
                         profile handed one scheme's bytes to another scheme's decoder",
                        got.scheme
                    );
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
