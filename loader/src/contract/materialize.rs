//! What materializing a checkpoint as pie's own format means.
//!
//! `pie model optimize` rewrites any checkpoint as a `.zt` artifact, and this
//! module derives the split that rewrite works from: which tensors *decode* —
//! a blocked scheme the host executor has a decoder for, undone to the logical
//! dtype no device kernel has to unpick — and which pass through byte for
//! byte, keeping their encoding. Quantized tensors without a decoder are in
//! the second set, not an error: `.zt` carries their scheme parametrically, so
//! copying them is exact.
//!
//! Everything here is derived from the checkpoint's own metadata — which
//! tensors exist and how each is encoded — and none of it from a model
//! family. That is what lets it live in the loader at all: the contract it
//! writes names no family convention. It only undoes an *encoding*, tensor by
//! tensor, under each tensor's own name, so the output is a checkpoint with
//! exactly the names and shapes a family contract already knows how to load.
//!
//! The contract covers the decoded set *only*. Passthrough tensors are copied
//! by the caller straight from the source files through a bounded buffer —
//! putting them in the contract would route gigabytes of untouched bytes
//! through the plan executor's memory to say "copy".

use crate::checkpoint::CheckpointMetadata;
use crate::contract::{Expr, ModelContract, TensorContract};
use crate::error::Result;
use crate::types::{Encoding, QuantScheme};

/// What materializing one checkpoint means, stated before it is done — the
/// shape a `--dry-run` reports.
pub struct Materialization {
    /// Decodes the blocked tensors; covers nothing else. Empty `tensors` when
    /// nothing decodes.
    pub contract: ModelContract,
    /// Tensors that decode (blocked scheme → logical dtype).
    pub decoded: Vec<String>,
    /// Tensors that pass through byte for byte, encoding and all.
    pub passthrough: Vec<String>,
}

/// Splits `metadata`'s tensors into decode and passthrough, and writes the
/// contract for the first set.
pub fn materialize_contract(metadata: &CheckpointMetadata) -> Result<Materialization> {
    let mut decoded = Vec::new();
    let mut passthrough = Vec::new();
    let mut tensors = Vec::new();
    for tensor in &metadata.tensors {
        match &tensor.encoding {
            // The blocked schemes the host executor decodes today. More move
            // up from the passthrough arm as their decoders land.
            Encoding::Quant(spec) if spec.scheme == QuantScheme::GgufQ4_0 => {
                decoded.push(tensor.name.clone());
                tensors.push(TensorContract::new(
                    &tensor.name,
                    Expr::src(&tensor.name).cast(Encoding::Raw(spec.logical_dtype)),
                    tensor.shape.clone(),
                    Encoding::Raw(spec.logical_dtype),
                ));
            }
            Encoding::Raw(_) | Encoding::Quant(_) => {
                passthrough.push(tensor.name.clone());
            }
        }
    }
    Ok(Materialization {
        contract: ModelContract {
            alignment: 1,
            tensors,
            groups: Vec::new(),
        },
        decoded,
        passthrough,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{CheckpointFile, CheckpointMetadata, RawTensor};
    use crate::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};
    use crate::types::{Axis, CheckpointFormat, DType, FileId, QuantSpec, TensorId};

    /// A mixed checkpoint — one Q4_0 tensor, one raw — splits into a decode
    /// plan for the first and a passthrough listing for the second, and the
    /// plan decodes end to end.
    #[test]
    fn a_gguf_checkpoint_materializes_to_plain_dtypes() {
        let dir = std::env::temp_dir().join(format!("pie_materialize_{}", std::process::id()));
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

        let materialization = materialize_contract(&metadata).unwrap();
        assert_eq!(materialization.decoded, ["w"]);
        assert_eq!(materialization.passthrough, ["bias"]);

        // The contract covers the decoded set only — the passthrough copy is
        // the caller's, straight from the file.
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let plan = crate::plan::compile(&metadata, &materialization.contract, target).unwrap();
        let storage = crate::testkit::host_executor::execute_plan(&plan, &dir).unwrap();

        let mut expected_w = Vec::new();
        for _ in 0..32 {
            expected_w.extend_from_slice(&half::bf16::from_f32(-16.0).to_bits().to_le_bytes());
        }
        assert_eq!(storage.tensors["w"], expected_w);
        assert!(!storage.tensors.contains_key("bias"));
        std::fs::remove_dir_all(dir).ok();
    }

    /// A checkpoint of raw tensors decodes nothing and copies everything.
    #[test]
    fn a_plain_checkpoint_is_all_passthrough() {
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
        let materialization = materialize_contract(&metadata).unwrap();
        assert!(materialization.decoded.is_empty());
        assert_eq!(materialization.passthrough, ["w"]);
        assert!(materialization.contract.tensors.is_empty());
    }

    /// A quantized tensor without a decoder is a copy, not an error — `.zt`
    /// carries its scheme parametrically.
    #[test]
    fn an_undecodable_quant_scheme_passes_through() {
        let metadata = CheckpointMetadata {
            files: vec![],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "w".to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: 512,
                shape: vec![1024],
                encoding: Encoding::Quant(QuantSpec {
                    scheme: QuantScheme::AwqInt4,
                    logical_dtype: DType::BF16,
                    bits_per_element: 4,
                    group_size: 128,
                    channel_axis: None,
                }),
            }],
        };
        let materialization = materialize_contract(&metadata).unwrap();
        assert!(materialization.decoded.is_empty());
        assert_eq!(materialization.passthrough, ["w"]);
    }
}
