//! What materializing a checkpoint as pie's own format means.
//!
//! `pie model convert` rewrites any checkpoint as a `.zt` artifact, and this
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
use crate::types::{DType, Encoding, QuantScheme};

/// What materializing one checkpoint means, stated before it is done — the
/// shape a `--dry-run` reports.
///
/// The three sets partition the source's objects exactly: every object is
/// decoded, copied, or carried metadata. Nothing falls off the end — a caller
/// that handles all three has handled the whole checkpoint.
pub struct Materialization {
    /// Decodes the blocked tensors; covers nothing else. Empty `tensors` when
    /// nothing decodes.
    pub contract: ModelContract,
    /// Tensors that decode (blocked scheme → logical dtype).
    pub decoded: Vec<String>,
    /// Tensors that pass through byte for byte, encoding and all.
    pub passthrough: Vec<String>,
    /// pie's own metadata objects, when the source is already a pie artifact.
    ///
    /// Called out as a set of its own rather than folded into `passthrough`
    /// because the decision about them is not the loader's: a re-convert that
    /// recompiles the tokenizer and the model descriptor wants to *drop* these
    /// and write fresh ones, while one that only re-lays the weights wants to
    /// carry them over. Both are right, and neither should happen by default
    /// because the loader forgot to mention them.
    pub meta: Vec<String>,
}

/// Splits `metadata`'s objects into decode, passthrough and metadata, and
/// writes the contract for the first set.
pub fn materialize_contract(metadata: &CheckpointMetadata) -> Result<Materialization> {
    let mut decoded = Vec::new();
    let mut passthrough = Vec::new();
    let mut tensors = Vec::new();
    let meta = metadata
        .meta_objects()
        .map(|tensor| tensor.name.clone())
        .collect();
    for tensor in metadata.weights() {
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
            // Float widths no kernel reads.
            //
            // Every device kernel pie ships reads BF16 -- norm weights, affine
            // scales and biases alike -- so a checkpoint that stores F16 or F32
            // is cast on the way in, by the CUDA driver and the Metal one
            // alike. That cast cannot be skipped and it cannot be a
            // reinterpretation: F16 and BF16 put the exponent in a different
            // place and at a different width, so reading one as the other turns
            // 0.0385 into 1.6e-12 without crashing or warning.
            //
            // Which makes it work this command exists to absorb. `.zt` is a
            // LOCAL artifact -- converted on the machine that will serve it,
            // for the engine that will serve it -- so the width every consumer
            // wants is a fact about the artifact, not a preference imposed on
            // it. mlx-community ships its affine scales and biases as F16, and
            // leaving them that way makes the driver rewrite them at every
            // boot; worse, it makes them the only tensors the driver cannot
            // bind where they lie, which on Qwen3.5-0.8B costs a 629 MB copy to
            // rewrite 0.0 MB.
            //
            // Narrowing, and deliberately: F16 carries three mantissa bits BF16
            // does not, and this drops them. It drops exactly the bits the
            // engine drops at load, so the artifact serves what a cold load
            // would have served -- which is this command's whole contract.
            Encoding::Raw(DType::F16) | Encoding::Raw(DType::F32) => {
                decoded.push(tensor.name.clone());
                tensors.push(TensorContract::new(
                    &tensor.name,
                    Expr::src(&tensor.name).cast(Encoding::Raw(DType::BF16)),
                    tensor.shape.clone(),
                    Encoding::Raw(DType::BF16),
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
        meta,
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

    /// An F16 tensor is rewritten as BF16, and the values survive the trip.
    ///
    /// The point is not that a cast works -- it is that the artifact carries
    /// the width the engine reads, so nothing is left for load time to do. The
    /// value is chosen to be exact in both widths: a check that passed on a
    /// reinterpretation instead of a conversion would prove nothing, and
    /// reinterpreting F16 `1.0` (0x3C00) as BF16 gives 0.0078125.
    #[test]
    fn an_f16_tensor_is_normalized_to_bf16() {
        let dir = std::env::temp_dir().join(format!("pie_f16norm_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.bin");
        let source: Vec<u8> = [1.0f32, -2.0, 0.5, 384.0]
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        std::fs::write(&path, &source).unwrap();

        let metadata = CheckpointMetadata {
            files: vec![CheckpointFile {
                id: FileId(0),
                path: "w.bin".to_string(),
                size_bytes: source.len() as u64,
                format: CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "w".to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: source.len() as u64,
                shape: vec![4],
                encoding: Encoding::Raw(DType::F16),
            }],
        };
        let materialization = materialize_contract(&metadata).unwrap();
        assert_eq!(materialization.decoded, ["w"]);
        assert!(materialization.passthrough.is_empty());

        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let plan = crate::plan::compile(&metadata, &materialization.contract, target).unwrap();
        let storage = crate::testkit::host_executor::execute_plan(&plan, &dir).unwrap();

        let expected: Vec<u8> = [1.0f32, -2.0, 0.5, 384.0]
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        assert_eq!(storage.tensors["w"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// A checkpoint of raw tensors the engine reads directly copies everything.
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
