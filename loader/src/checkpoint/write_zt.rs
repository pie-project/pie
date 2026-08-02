//! Writing a `.zt` checkpoint: the output side of `convert` and `optimize`.
//!
//! The counterpart to [`write_safetensors`](super::write::write_safetensors),
//! and the difference between them is what the container can say. safetensors
//! describes a tensor as a dtype tag over a shape, so a payload it has no tag
//! for cannot be written at all — which is why `convert` refuses every
//! quantized output outside {MXFP4, FP8, INT8}. A `.zt` object carries a
//! *layout*, so the scheme names itself and any scheme the loader can describe
//! can be written down.
//!
//! Three things come free with the format and are the reason to prefer it for
//! artifacts pie itself produces:
//!
//! - **Alignment.** Canonical placement puts every tensor on a 64 KiB
//!   boundary, so a written artifact is already streamable and does not need
//!   the [`align`](super::align) rewrite afterwards — nor the filler tensors
//!   that rewrite has to invent to stay inside safetensors.
//! - **Integrity.** Every tensor carries an XXH3 digest and the manifest
//!   carries one of its own, so a cache that rots is an error at load rather
//!   than a wrong answer.
//! - **Provenance.** What the artifact was derived from goes in the file's
//!   attributes instead of a sidecar or a directory name.

use std::collections::BTreeMap;
use std::path::Path;

use ztensor::cbor::Value;
use ztensor::{DType as ZDType, PartDef};

use crate::checkpoint::write::WriteTensor;
use crate::error::Error;
use crate::types::{DType, Encoding, QuantScheme};

/// The storage type and logical type a declaration's elements are stored as.
///
/// A quantized payload is bytes: the scheme says how to read them, and the
/// layout on the object says which scheme. The one exception is MXFP4's
/// payload, whose elements are half a byte — `f4_e2m1` says so, and the size
/// equation the reader checks follows from it.
fn storage_of(decl_dtype: DType, encoding: &Encoding) -> Result<(ZDType, Option<&'static str>), Error> {
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

/// The layout profile a declaration's encoding names.
fn layout_of(encoding: &Encoding) -> Result<&'static str, Error> {
    let Encoding::Quant(spec) = encoding else {
        return Ok("dense");
    };
    Ok(match spec.scheme {
        QuantScheme::None => "dense",
        QuantScheme::Mxfp4E2M1E8M0 => "zt.mx/1",
        QuantScheme::GgufQ4_0 => "gguf.q4_0/1",
        QuantScheme::GgufQ4K => "gguf.q4_k/1",
        QuantScheme::GgufQ5_0 => "gguf.q5_0/1",
        QuantScheme::GgufQ5K => "gguf.q5_k/1",
        QuantScheme::GgufQ8_0 => "gguf.q8_0/1",
        // The affine-integer family: bits, group size and packing are what
        // distinguish them, and the profile carries all three.
        QuantScheme::AwqInt4
        | QuantScheme::GptqInt4
        | QuantScheme::MlxAffineU4
        | QuantScheme::Int4B8
        | QuantScheme::Int8Symmetric
        | QuantScheme::Int8Asymmetric
        | QuantScheme::Fp8E4M3
        | QuantScheme::Fp8E5M2 => "zt.quant_group/1",
    })
}

/// The attributes a quantized object needs for its scheme to be readable
/// again: what a reader must know that the shape does not say.
fn attributes_of(encoding: &Encoding) -> Option<Value> {
    let Encoding::Quant(spec) = encoding else {
        return None;
    };
    let spec = spec.clone().normalized();
    let mut entries = vec![
        (
            Value::Text("bits".into()),
            Value::Uint(u64::from(spec.bits_per_element)),
        ),
        (
            Value::Text("group_size".into()),
            Value::Uint(u64::from(spec.group_size)),
        ),
        (
            Value::Text("scheme".into()),
            Value::Text(format!("{:?}", spec.scheme)),
        ),
    ];
    if let Some(axis) = spec.channel_axis {
        entries.push((Value::Text("axis".into()), Value::Uint(u64::from(axis.0))));
    }
    Some(Value::Map(entries))
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
    let zt_err = |err: ztensor::Error| Error::Checkpoint(err.to_string());

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|err| Error::Checkpoint(format!("cannot create {}: {err}", parent.display())))?;
    }
    // Publication is atomic, matching `write_safetensors`: a run that dies
    // mid-write leaves no file that parses.
    let temporary = path.with_extension(format!("zt.{}.partial", std::process::id()));
    let mut writer = ztensor::Writer::create(&temporary).map_err(zt_err)?;

    if !metadata.is_empty() {
        writer.set_attributes(Value::Map(
            metadata
                .iter()
                .map(|(k, v)| (Value::Text(k.clone()), Value::Text(v.clone())))
                .collect(),
        ));
    }

    let mut ordered: Vec<&WriteTensor<'_>> = tensors.iter().collect();
    ordered.sort_by(|a, b| a.decl.name.cmp(&b.decl.name));

    for tensor in ordered {
        let decl = tensor.decl;
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
        writer
            .add_object(
                &decl.name,
                &shape,
                layout_of(&decl.encoding)?,
                &[(
                    "data",
                    PartDef {
                        dtype,
                        ltype,
                        encoding: None,
                        data: tensor.bytes,
                    },
                )],
                attributes_of(&decl.encoding),
            )
            .map_err(zt_err)?;
    }
    writer.finish().map_err(zt_err)?;

    std::fs::rename(&temporary, path).map_err(|err| {
        let _ = std::fs::remove_file(&temporary);
        Error::Checkpoint(format!("cannot publish {}: {err}", path.display()))
    })?;
    Ok(())
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
