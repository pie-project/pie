//! Writing a safetensors checkpoint: the output side of `convert`.
//!
//! The one place the loader authors a header rather than preserving one.
//! `align` rewrites a checkpoint and keeps every header entry byte for byte;
//! this module is for tensors that did not exist until a plan ran — a weight
//! quantized on the host, and the scale tensor the encode published beside it.
//!
//! What it writes is a checkpoint any safetensors reader accepts: tensors
//! back to back, offsets contiguous, dtype tags from the reference
//! implementation's vocabulary. An MXFP4 payload is tagged `F4_E2M1` over its
//! *logical* shape — two elements per byte, the arithmetic `read.rs` checks on
//! the way back in — and its scales stay the plain `U8` their declaration
//! carries, because a converted checkpoint that only this repository could
//! read would be a new format wearing a familiar extension (`align.rs` says
//! the same about gaps).
//!
//! Publication is atomic: bytes go to a process-unique temporary and arrive by
//! `rename`, so a run that dies mid-write leaves no file that parses. The rule
//! that the destination must not already exist is the caller's, who knows what
//! the destination is for; this module only refuses to tear a file.

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::Error;
use crate::types::{DType, Encoding, QuantScheme, TensorDecl};

/// One tensor of the file: what to call it, and the bytes as stored.
pub struct WriteTensor<'a> {
    pub decl: &'a TensorDecl,
    pub bytes: &'a [u8],
}

/// The safetensors dtype tag for a declaration, and the byte count its shape
/// implies — stated together because the pair is the whole contract between
/// tag and payload: `F4_E2M1` is the one tag whose element is half a byte.
fn tag_and_span(decl: &TensorDecl) -> Result<(&'static str, u64), Error> {
    let elements: i64 = decl.shape.iter().product();
    if decl.shape.iter().any(|&extent| extent < 0) || elements < 0 {
        return Err(Error::Checkpoint(format!(
            "tensor {} has negative shape {:?}",
            decl.name, decl.shape
        )));
    }
    let elements = elements as u64;
    match &decl.encoding {
        Encoding::Raw(dtype) => {
            let tag = match dtype {
                DType::F32 => "F32",
                DType::F16 => "F16",
                DType::BF16 => "BF16",
                DType::F8E4M3 => "F8_E4M3",
                DType::F8E5M2 => "F8_E5M2",
                // Written under the storage tag it will be read back as:
                // `read.rs` maps `F8_E8M0` to `U8` on purpose, and a contract
                // that decodes says so with `Transmute`.
                DType::E8M0 => "F8_E8M0",
                DType::I64 => "I64",
                DType::I32 => "I32",
                DType::I16 => "I16",
                DType::I8 => "I8",
                DType::U64 => "U64",
                DType::U32 => "U32",
                DType::U16 => "U16",
                DType::U8 => "U8",
                DType::Bool => "BOOL",
            };
            Ok((tag, elements * dtype.bytes()))
        }
        Encoding::Quant(spec) => match spec.scheme {
            QuantScheme::Mxfp4E2M1E8M0 => {
                if !elements.is_multiple_of(2) {
                    return Err(Error::Checkpoint(format!(
                        "tensor {} is F4_E2M1 with an odd element count {elements}",
                        decl.name
                    )));
                }
                Ok(("F4_E2M1", elements / 2))
            }
            // Byte-wide schemes store one element per byte under the raw tag
            // of their storage type; what makes them quantized — the factor
            // tensor beside them — is the contract's statement, not the tag's.
            QuantScheme::Fp8E4M3 => Ok(("F8_E4M3", elements)),
            QuantScheme::Int8Symmetric => Ok(("I8", elements)),
            other => Err(Error::Checkpoint(format!(
                "tensor {}: safetensors has no tag for {other:?}",
                decl.name
            ))),
        },
    }
}

/// Write `tensors`, in order, as one safetensors file at `path`.
///
/// `metadata` lands under `__metadata__` when non-empty — provenance, not
/// meaning; nothing this crate reads depends on it.
pub fn write_safetensors(
    path: &Path,
    metadata: &BTreeMap<String, String>,
    tensors: &[WriteTensor<'_>],
) -> Result<(), Error> {
    let cannot = |what: &str, err: std::io::Error| {
        Error::Checkpoint(format!("cannot {what} {}: {err}", path.display()))
    };

    // Offsets first, so the header is complete before anything is written.
    let mut entries = Vec::with_capacity(tensors.len());
    let mut at = 0u64;
    for tensor in tensors {
        let (tag, span) = tag_and_span(tensor.decl)?;
        if tensor.bytes.len() as u64 != span {
            return Err(Error::Checkpoint(format!(
                "tensor {} has {} bytes but its {tag} shape {:?} needs {span}",
                tensor.decl.name,
                tensor.bytes.len(),
                tensor.decl.shape
            )));
        }
        entries.push((tensor, tag, at, at + span));
        at += span;
    }

    // The header by hand, in entry order. `serde_json`'s map would sort the
    // names, which is legal safetensors but would shuffle the file away from
    // the order the caller chose — and the caller's order is the plan's, which
    // is the one a reader diffing two conversions wants to see preserved.
    let mut header = String::from("{");
    if !metadata.is_empty() {
        header.push_str("\"__metadata__\":{");
        for (index, (key, value)) in metadata.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            header.push_str(&serde_json::to_string(key).expect("a string serializes"));
            header.push(':');
            header.push_str(&serde_json::to_string(value).expect("a string serializes"));
        }
        header.push('}');
        if !entries.is_empty() {
            header.push(',');
        }
    }
    for (index, (tensor, tag, begin, end)) in entries.iter().enumerate() {
        if index > 0 {
            header.push(',');
        }
        header.push_str(&serde_json::to_string(&tensor.decl.name).expect("a string serializes"));
        header.push_str(&format!(
            ":{{\"dtype\":\"{tag}\",\"shape\":{:?},\"data_offsets\":[{begin},{end}]}}",
            tensor.decl.shape
        ));
    }
    header.push('}');

    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(directory).map_err(|err| cannot("create directory for", err))?;
    let temporary = directory.join(format!(
        ".{}.pie-write.{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("checkpoint"),
        std::process::id()
    ));
    let write = || -> std::io::Result<()> {
        let file = std::fs::File::create(&temporary)?;
        let mut out = BufWriter::new(file);
        out.write_all(&(header.len() as u64).to_le_bytes())?;
        out.write_all(header.as_bytes())?;
        for (tensor, ..) in &entries {
            out.write_all(tensor.bytes)?;
        }
        out.into_inner()?.sync_all()
    };
    if let Err(err) = write() {
        std::fs::remove_file(&temporary).ok();
        return Err(cannot("write", err));
    }
    std::fs::rename(&temporary, path).map_err(|err| {
        std::fs::remove_file(&temporary).ok();
        cannot("publish", err)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Axis, QuantSpec, TensorId, Visibility};

    fn decl(name: &str, shape: Vec<i64>, encoding: Encoding) -> TensorDecl {
        TensorDecl {
            id: TensorId(0),
            name: name.to_string(),
            shape,
            encoding,
            alignment: 1,
            visibility: Visibility::Public,
        }
    }

    /// The file this writes is read back by this crate's own parser, which is
    /// the arithmetic that matters: `F4_E2M1` spans half its element count,
    /// and an entry that lied about it would fail `read.rs`'s span check.
    #[test]
    fn a_written_checkpoint_parses_back_with_the_same_layout() {
        let dir = std::env::temp_dir().join(format!("pie_write_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let payload = decl(
            "w",
            vec![2, 32],
            Encoding::Quant(QuantSpec {
                scheme: QuantScheme::Mxfp4E2M1E8M0,
                logical_dtype: DType::BF16,
                bits_per_element: 4,
                group_size: 32,
                channel_axis: Some(Axis(1)),
            }),
        );
        let scales = decl("w_scale", vec![2, 2], Encoding::Raw(DType::U8));
        let payload_bytes: Vec<u8> = (0..32).collect();
        let scale_bytes = vec![127u8, 128, 126, 129];
        let mut metadata = BTreeMap::new();
        metadata.insert("pie_convert".to_string(), "test".to_string());

        write_safetensors(
            &dir.join("model.safetensors"),
            &metadata,
            &[
                WriteTensor {
                    decl: &payload,
                    bytes: &payload_bytes,
                },
                WriteTensor {
                    decl: &scales,
                    bytes: &scale_bytes,
                },
            ],
        )
        .unwrap();

        let parsed = crate::checkpoint::read::parse_checkpoint_metadata(&dir).unwrap();
        assert_eq!(parsed.tensors.len(), 2);
        let w = parsed.tensor_by_name("w").unwrap();
        assert_eq!(w.shape, vec![2, 32]);
        assert_eq!(w.span_bytes, 32);
        let s = parsed.tensor_by_name("w_scale").unwrap();
        assert_eq!(s.shape, vec![2, 2]);
        assert_eq!(s.span_bytes, 4);
        assert_eq!(s.file_offset, w.file_offset + 32);
        std::fs::remove_dir_all(dir).ok();
    }

    /// A payload byte count that disagrees with the declared shape is refused
    /// before anything lands on disk.
    #[test]
    fn a_span_mismatch_is_refused_and_writes_nothing() {
        let dir = std::env::temp_dir().join(format!("pie_write_bad_{}", std::process::id()));
        let tensor = decl("w", vec![4], Encoding::Raw(DType::F32));
        let short = vec![0u8; 12];
        let result = write_safetensors(
            &dir.join("model.safetensors"),
            &BTreeMap::new(),
            &[WriteTensor {
                decl: &tensor,
                bytes: &short,
            }],
        );
        assert!(result.is_err());
        assert!(!dir.join("model.safetensors").exists());
        std::fs::remove_dir_all(dir).ok();
    }
}
