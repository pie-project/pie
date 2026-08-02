//! The family-agnostic normalization contract.
//!
//! Everything here is derived from the checkpoint's own metadata — which
//! tensors exist and how each is encoded — and none of it from a model
//! family. That is what lets it live in the loader at all: the contract it
//! writes names no family convention. It only undoes an *encoding* no device
//! kernel reads, tensor by tensor, under each tensor's own name, so the
//! output is a checkpoint with exactly the names and shapes a family contract
//! already knows how to load.
//!
//! Today that means GGUF: a Q4_0 tensor decodes to its logical dtype and
//! every raw tensor passes through unchanged. More blocked schemes slot into
//! the same match as their decoders land in the host executor.

use crate::checkpoint::CheckpointMetadata;
use crate::contract::{Expr, ModelContract, TensorContract};
use crate::error::{Error, Result};
use crate::types::{Encoding, QuantScheme};

/// What normalizing one checkpoint means, stated before it is done — the
/// shape a `--dry-run` reports.
pub struct Normalization {
    pub contract: ModelContract,
    /// Tensors that decode (blocked scheme → logical dtype).
    pub decoded: Vec<String>,
    /// Tensors that pass through byte for byte.
    pub passthrough: Vec<String>,
}

/// The contract that rewrites `metadata`'s checkpoint into plain dtypes, or
/// `None` when nothing in it is encoded — a checkpoint of raw tensors is
/// already its own normalization, and rewriting it would copy gigabytes to
/// say so.
pub fn normalize_contract(metadata: &CheckpointMetadata) -> Result<Option<Normalization>> {
    let mut decoded = Vec::new();
    let mut passthrough = Vec::new();
    let mut tensors = Vec::with_capacity(metadata.tensors.len());
    for tensor in &metadata.tensors {
        match &tensor.encoding {
            Encoding::Raw(dtype) => {
                passthrough.push(tensor.name.clone());
                tensors.push(TensorContract::new(
                    &tensor.name,
                    Expr::src(&tensor.name),
                    tensor.shape.clone(),
                    Encoding::Raw(*dtype),
                ));
            }
            Encoding::Quant(spec) if spec.scheme == QuantScheme::GgufQ4_0 => {
                decoded.push(tensor.name.clone());
                tensors.push(TensorContract::new(
                    &tensor.name,
                    Expr::src(&tensor.name).cast(Encoding::Raw(spec.logical_dtype)),
                    tensor.shape.clone(),
                    Encoding::Raw(spec.logical_dtype),
                ));
            }
            Encoding::Quant(spec) => {
                return Err(Error::Checkpoint(format!(
                    "tensor '{}' is {:?}, which normalization has no decoder for",
                    tensor.name, spec.scheme
                )));
            }
        }
    }
    if decoded.is_empty() {
        return Ok(None);
    }
    Ok(Some(Normalization {
        contract: ModelContract {
            alignment: 1,
            tensors,
            groups: Vec::new(),
        },
        decoded,
        passthrough,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
    use crate::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
    use crate::types::{Axis, CheckpointFormat, DType, FileId, QuantSpec, TensorId};

    /// A mixed checkpoint — one Q4_0 tensor, one raw — normalizes to a plan
    /// that decodes the first and copies the second, end to end.
    #[test]
    fn a_gguf_checkpoint_normalizes_to_plain_dtypes() {
        let dir = std::env::temp_dir().join(format!("pie_normalize_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // One Q4_0 block (scale 2.0, all-zero codes: every nibble is 0 − 8,
        // so all 32 elements decode to −16.0) and four raw BF16 values
        // behind it.
        let mut file = Vec::new();
        file.extend_from_slice(&[0x00, 0x40]);
        file.extend_from_slice(&[0x00; 16]);
        for value in [1.0f32, 2.0, 3.0, 4.0] {
            file.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
        }
        std::fs::write(dir.join("model.gguf"), &file).unwrap();

        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "model.gguf".to_string(),
                size_bytes: 26,
                format: CheckpointFormat::Gguf,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: 0,
                    span_bytes: 18,
                    shape: vec![32],
                    encoding: Encoding::Quant(QuantSpec {
                        scheme: QuantScheme::GgufQ4_0,
                        logical_dtype: DType::BF16,
                        bits_per_element: 4,
                        group_size: 32,
                        channel_axis: Some(Axis(0)),
                    }),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "bias".to_string(),
                    file_id: FileId(0),
                    file_offset: 18,
                    span_bytes: 8,
                    shape: vec![4],
                    encoding: Encoding::Raw(DType::BF16),
                },
            ],
        };

        let normalization = normalize_contract(&metadata).unwrap().unwrap();
        assert_eq!(normalization.decoded, ["w"]);
        assert_eq!(normalization.passthrough, ["bias"]);

        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let plan = crate::plan::compile(&metadata, &normalization.contract, target).unwrap();
        let storage = crate::testkit::host_executor::execute_plan(&plan, &dir).unwrap();

        let mut expected_w = Vec::new();
        for _ in 0..32 {
            expected_w.extend_from_slice(&half::bf16::from_f32(-16.0).to_bits().to_le_bytes());
        }
        assert_eq!(storage.tensors["w"], expected_w);
        assert_eq!(storage.tensors["bias"], file[18..26]);
        std::fs::remove_dir_all(dir).ok();
    }

    /// A checkpoint of raw tensors has nothing to normalize, and says so
    /// rather than copying it.
    #[test]
    fn a_plain_checkpoint_normalizes_to_nothing() {
        let metadata = CheckpointMetadata {
            files: vec![],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "w".to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: 8,
                shape: vec![4],
                encoding: Encoding::Raw(DType::BF16),
            }],
        };
        assert!(normalize_contract(&metadata).unwrap().is_none());
    }
}
