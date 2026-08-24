//! What materializing a checkpoint as pie's own format means.
//!
//! `pie model import` rewrites any checkpoint as a `.zt` artifact, and this
//! module derives the split that rewrite works from: which tensors are
//! *rewritten* — narrowed to the BF16 every device kernel reads — and which
//! pass through byte for byte, keeping the encoding they arrived in.
//!
//! **What is NOT in the first set, and used to be.** A self-contained block —
//! every `Gguf*`, whose scales are interleaved with its codes — passes
//! through. Unpacking one here made an archive a BF16 thing, which was never a
//! decision: it was the only way to be sure the archive could be served, back
//! when nothing downstream could read a block. `986252f35` moved that decode
//! to the point that needs the tensor unpacked, and an archive is now 3.6x
//! smaller and several times faster to write. So `decoded` today means F16 or
//! F32 narrowed to BF16, and nothing else. Quantized tensors are all in the
//! second set, which is not an error: `.zt` carries their scheme
//! parametrically, so copying them is exact.
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
use crate::types::{DType, Encoding};

/// What materializing one checkpoint means, stated before it is done — the
/// shape a `--dry-run` reports.
///
/// The three sets partition the source's objects exactly: every object is
/// rewritten, copied, or carried metadata. Nothing falls off the end — a
/// caller that handles all three has handled the whole checkpoint.
pub struct Materialization {
    /// Covers the rewritten set and nothing else. Empty `tensors` when nothing
    /// is rewritten, which is the common case: a BF16 checkpoint copies whole,
    /// and so now does a GGUF.
    pub contract: ModelContract,
    /// Tensors rewritten on the way in — as this function leaves it, F16 or
    /// F32 narrowed to BF16 and nothing else.
    ///
    /// "As this function leaves it" is load-bearing: `pie model import`
    /// PROMOTES into this set afterwards. A tensor that needs a transform the
    /// checkpoint's vocabulary asks for — llama's Q/K row permutation, gemma's
    /// unfolded norms — cannot be a byte copy, so import moves it out of
    /// `passthrough` and adds it here, keeping whatever encoding it had.
    /// Since blocks began surviving import, some of those are blocked.
    ///
    /// The name is older than the meaning. It once also held every blocked
    /// tensor, because import unpacked them; since it stopped, this set is
    /// narrowings alone. Anything reporting on it should say so — a count of
    /// this set described as "blocked tensors decoded" was true for years and
    /// then quietly was not.
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

/// Splits `metadata`'s objects into rewrite, passthrough and metadata, and
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
            // The blocked schemes the host executor decodes. The condition is
            // the executor's own admission test rather than a list repeated
            // here, so a scheme cannot be decodable in one file and opaque in
            // the other.
            //
            // **These are NO LONGER decoded here.** A block whose scales live
            // inside it passes through packed, and is decoded at the point
            // that needs it unpacked -- see the `is_none()` arm below. This
            // paragraph records why, because the opposite was true for most of
            // this file's life and the reasoning is not recoverable from the
            // diff.
            //
            // Decoding on the way in made an archive a BF16 thing. That was
            // never a decision; it was the only way to be sure the archive
            // could be served, back when nothing downstream could read a
            // block. Three commits removed that constraint: the author decodes
            // a bound block (`Builder::decode_bound_blocks`), the device masks
            // admit `TILE_MAP_DECODE`, and `validate_bound_encodings` refuses
            // the case where neither happened. A fourth, `ab3682829`, closed
            // the hole that made the change unsafe -- every alignment rule in
            // `infer` was keyed on `channel_axis`, which these schemes do not
            // state, so a `Stride` across a block was silently ACCEPTED. It is
            // now grouped along its fastest axis like any other quantized
            // tensor.
            //
            // Measured, `Llama-3.2-1B-Instruct-Q3_K_M`:
            //
            // | | archive | import | build (cold) | resident |
            // |---|---|---|---|---|
            // | decoded | 2.3 GiB | 1 s | 2.93 s | 2621 MiB |
            // | preserved | **657 MiB** | **346 ms** | **1.90 s** | 2621 MiB |
            //
            // Note the build column: the packed archive is faster to build
            // FROM, decode included, because 3.6x less I/O more than pays for
            // the arithmetic. And on `qwen2.5-0.5b-instruct-q4_0` the two
            // routes' built runtimes agree on all 290 tensors byte for byte --
            // where the decode happens does not change the answer.
            //
            // The cost is a host decode at every boot taken FROM THE ARCHIVE
            // rather than from a built runtime. Measured cold, that boot is
            // not slower (0.22-0.23 s against 0.22-0.27 s): the decode is
            // cheaper than the I/O it avoids. An offline `pie model build`
            // WAS how the price is paid once regardless -- that command is
            // deleted, so today every boot from a packed archive pays this
            // decode, and the measurement above is why that is affordable.
            //
            // Swept both ways before flipping -- Llama-3.2-1B at IQ4_XS, Q2_K,
            // UD-IQ2_XXS and UD-Q2_K_XL, qwen2.5-0.5b Q4_K_M, the two-shard
            // qwen2.5-7b Q4_0, and gpt-oss-20b MXFP4 -- and nothing refused
            // that did not already refuse identically when decoded.
            Encoding::Quant(spec) if spec.scheme.is_self_contained() => {
                passthrough.push(tensor.name.clone());
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
    use crate::types::{Axis, CheckpointFormat, DType, FileId, QuantScheme, QuantSpec, TensorId};

    /// A mixed checkpoint — one Q4_0 tensor, one raw — carries BOTH through
    /// byte for byte, and the block it kept still decodes to the right values.
    ///
    /// The first half is the flip: a self-contained block is no longer
    /// unpacked on the way into an archive, so it needs no contract and joins
    /// the raw tensor in `passthrough`. The second half is why that is safe to
    /// do — the bytes are preserved, not merely copied, so a later consumer
    /// asking for BF16 gets exactly what an import-time decode would have
    /// written. Both are asserted here because either alone would pass while
    /// the other was broken: a passthrough that dropped the scale would still
    /// be listed, and a decode that worked would prove nothing about what the
    /// archive holds.
    #[test]
    fn a_block_reaches_the_archive_packed_and_still_decodes() {
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
        assert!(
            materialization.decoded.is_empty(),
            "a block is kept as it was stored, so nothing is decoded on the way in"
        );
        assert_eq!(materialization.passthrough, ["w", "bias"]);
        assert!(
            materialization.contract.tensors.is_empty(),
            "with nothing to transform there is no contract to write"
        );

        // What the archive keeps is still a Q4_0 block and not just 18 opaque
        // bytes: ask for the decode the import used to do, and the values come
        // back the same. The scale is in byte 0-1 and the codes follow, so a
        // passthrough that lost or reordered either would read differently.
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("w").cast(Encoding::Raw(DType::BF16)),
                vec![32],
                Encoding::Raw(DType::BF16),
            )],
            groups: Vec::new(),
        };
        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        let storage = crate::executor::Execution::new(&plan, &dir).run().unwrap();

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
        let storage = crate::executor::Execution::new(&plan, &dir).run().unwrap();

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

    /// A planar quantized tensor is a copy, not an error — `.zt` carries its
    /// scheme parametrically.
    ///
    /// AWQ keeps its scales in a plane of their own, so there is nothing to
    /// unpack even in principle. It reaches passthrough by a different route
    /// than a `Gguf*` block does — that one is decodable here and is kept
    /// anyway — and both routes have to arrive.
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
