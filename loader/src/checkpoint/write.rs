//! Writing a checkpoint: the output side of `convert` and `optimize`.
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
use std::path::Path;

use ztensor::DType as ZDType;
use ztensor::cbor::Value;

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
            Some(Value::Map(vec![
                (Value::Text("elems_per_block".into()), Value::Uint(elems)),
                (Value::Text("block_bytes".into()), Value::Uint(bytes)),
            ])),
        ));
    }

    match spec.scheme {
        QuantScheme::None => Ok(("dense", None)),

        // OCP Microscaling: its own profile, because the element is a
        // sub-byte float and the scale form is fixed by that specification.
        QuantScheme::Mxfp4E2M1E8M0 => Ok((
            "zt.mx/1",
            Some(Value::Map(vec![
                (Value::Text("axis".into()), Value::Uint(axis)),
                (Value::Text("block_size".into()), Value::Uint(group)),
                (
                    Value::Text("scale_form".into()),
                    Value::Text("e8m0_exponent".into()),
                ),
            ])),
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
                Some(Value::Map(vec![
                    (Value::Text("axis".into()), Value::Uint(axis)),
                    (Value::Text("bits".into()), Value::Uint(bits)),
                    (Value::Text("group_size".into()), Value::Uint(group)),
                    (
                        Value::Text("packing".into()),
                        Value::Map(vec![
                            (Value::Text("order".into()), Value::Text(order.into())),
                            (Value::Text("per_word".into()), Value::Uint(per_word)),
                            (Value::Text("word".into()), Value::Text(word.into())),
                        ]),
                    ),
                    (
                        Value::Text("scale_form".into()),
                        Value::Text(scale_form.into()),
                    ),
                    (Value::Text("zero_point".into()), zero),
                ])),
            ))
        }
    }
}

fn zero_none() -> Value {
    Value::Map(vec![(
        Value::Text("form".into()),
        Value::Text("none".into()),
    )])
}

fn zero_implied(value: u64) -> Value {
    Value::Map(vec![
        (Value::Text("form".into()), Value::Text("implied".into())),
        (Value::Text("value".into()), Value::Uint(value)),
    ])
}

fn zero_tensor(packing: &str) -> Value {
    Value::Map(vec![
        (Value::Text("form".into()), Value::Text("tensor".into())),
        (Value::Text("packing".into()), Value::Text(packing.into())),
    ])
}

/// Writes a checkpoint one tensor at a time, payloads in chunks.
///
/// The streaming face of [`write_zt`], for callers whose payloads should never
/// all be resident at once — `pie model optimize` copies multi-gigabyte
/// tensors straight from the source checkpoint through a bounded buffer.
/// Canonical form requires objects in ascending name order; [`write_zt`] sorts
/// for its caller, this type trusts its caller to add in order.
pub struct CheckpointWriter {
    /// `None` only after [`finish`](CheckpointWriter::finish) has taken it.
    writer: Option<ztensor::Writer>,
    open: Option<ztensor::Sink>,
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
        })
    }

    /// Declares a tensor and opens it for writing. Its payload is exactly
    /// `nbytes` bytes, delivered by [`write`](Self::write).
    pub fn begin_tensor(&mut self, decl: &TensorDecl, nbytes: u64) -> Result<(), Error> {
        if self.open.is_some() {
            return Err(Error::Checkpoint(format!(
                "tensor {} was begun while another is still open",
                decl.name
            )));
        }
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
        let mut object = self.writer().object(&decl.name).shape(shape).layout(layout);
        if let Some(attributes) = attributes {
            object = object.attributes(attributes);
        }
        object = object.part("data").dtype(dtype);
        if let Some(ltype) = ltype {
            object = object.logical(ltype);
        }
        self.open = Some(object.length(nbytes).stream().map_err(Error::from)?);
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

    /// Closes the manifest and moves the file into place.
    pub fn finish(mut self) -> Result<(), Error> {
        if self.open.is_some() {
            return Err(Error::Checkpoint(
                "finish was called while a tensor is still open".into(),
            ));
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

    #[test]
    fn dense_tensors_round_trip_through_the_reader() {
        let dir = tmpdir("dense");
        let path = dir.join("model.zt");
        let a: Vec<u8> = (0..32u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let b = vec![7u8; 16];
        let da = decl("a.weight", vec![8, 4], Encoding::Raw(DType::F32));
        let db = decl("b.mask", vec![16], Encoding::Raw(DType::U8));
        let mut meta = BTreeMap::new();
        meta.insert("pie_optimize".into(), "normalize".into());

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
